from convolution import Network
import numpy as np
from sys import stdout
import time

from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.integrate import trapz

def train_net(net):

    path         = net.path
    hidden_sizes = net.hyperparameters["hidden_sizes"]
    n_epochs     = net.hyperparameters["n_epochs"]
    batch_size   = net.hyperparameters["batch_size"]
    n_it_neg     = net.hyperparameters["n_it_neg"]
    n_it_pos     = net.hyperparameters["n_it_pos"]
    epsilon      = net.hyperparameters["epsilon"]
    beta         = net.hyperparameters["beta"]
    alphas       = net.hyperparameters["alphas"]

    print "name = %s" % (path)
    print "architecture = 784-"+"-".join([str(n) for n in hidden_sizes])+"-10"
    print "number of epochs = %i" % (n_epochs)
    print "batch_size = %i" % (batch_size)
    print "n_it_neg = %i"   % (n_it_neg)
    print "n_it_pos = %i"   % (n_it_pos)
    print "epsilon = %.1f" % (epsilon)
    print "beta = %.1f" % (beta)
    print "learning rates: "+" ".join(["alpha_W%i=%.3f" % (k+1,alpha) for k,alpha in enumerate(alphas)])+"\n"

    # SELECT TOTAL SIZE OF DATASETS (10000 TRAINING, 2000 VALIDATION) AND NUMBER OF BATCHES
    n_batches_train = 10000 / batch_size
    n_batches_valid = 2000 / batch_size

    start_time = time.clock()

    # CUMULATIVE SUM OF TRAINING ENERGY, TRAINING COST AND TRAINING ERROR
    measures_sum = [0.,0.,0.]

    for index in xrange(n_batches_train):

        # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
        net.change_mini_batch_index(index)

        # FREE PHASE
        net.free_phase(n_it_neg, epsilon)
        
        # MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
        measures = net.measure()
        measures_sum = [measure_sum + measure for measure_sum,measure in zip(measures_sum,measures)]
        measures_avg = [measure_sum / (index+1) for measure_sum in measures_sum]
        measures_avg[-1] *= 100. # measures_avg[-1] corresponds to the error rate, which we want in percentage
        stdout.write("\r   valid-%5i E=%.1f C=%.5f error=%.2f%%" % ((index+1)*batch_size, measures_avg[0], measures_avg[1], measures_avg[2]))
        stdout.flush()

        net.training_curves["training error"].append(measures[-1]*100)

    stdout.write("\n")

    # CUMULATIVE SUM OF VALIDATION ENERGY, VALIDATION COST AND VALIDATION ERROR
    measures_sum = [0.,0.,0.]

    for index in xrange(n_batches_valid):

        # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
        net.change_mini_batch_index(n_batches_train+index)

        # FREE PHASE
        net.free_phase(n_it_neg, epsilon)
        
        # MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
        measures = net.measure()
        measures_sum = [measure_sum + measure for measure_sum,measure in zip(measures_sum,measures)]
        measures_avg = [measure_sum / (index+1) for measure_sum in measures_sum]
        measures_avg[-1] *= 100. # measures_avg[-1] corresponds to the error rate, which we want in percentage
        stdout.write("\r   valid-%5i E=%.1f C=%.5f error=%.2f%%" % ((index+1)*batch_size, measures_avg[0], measures_avg[1], measures_avg[2]))
        stdout.flush()

        net.training_curves["validation error"].append(measures[-1]*100)

    stdout.write("\n")

    for epoch in range(n_epochs):

        ### TRAINING ###

        # CUMULATIVE SUM OF TRAINING ENERGY, TRAINING COST AND TRAINING ERROR
        measures_sum = [0.,0.,0.]
        gW = [0.] * len(alphas)

        for index in xrange(n_batches_train):

            # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
            net.change_mini_batch_index(index)

            # FREE PHASE
            net.free_phase(n_it_neg, epsilon)

            # MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
            measures = net.measure()
            measures_sum = [measure_sum + measure for measure_sum,measure in zip(measures_sum,measures)]
            measures_avg = [measure_sum / (index+1) for measure_sum in measures_sum]
            measures_avg[-1] *= 100. # measures_avg[-1] corresponds to the error rate, which we want in percentage
            stdout.write("\r%2i-train-%5i E=%.1f C=%.5f error=%.3f%%" % (epoch, (index+1)*batch_size, measures_avg[0], measures_avg[1], measures_avg[2]))
            stdout.flush()

            net.training_curves["training error"].append(measures[-1]*100)

            # WEAKLY CLAMPED PHASE
            dparams_history = []

            # SET THE NUMBER OF INTERMEDIATE BETA VALUES ON EACH OF THE SAWTOOTH RAMPS
            pause_num = 9

            # SIGN OF CLAMPING IS CHOSEN RANDOMLY
            sign = 2*np.random.randint(0,2)-1
            beta = sign*beta

            # FOR EACH INTERMEDIATE BETA VALUE WE RECORD THE
            for i in np.arange(pause_num):

                # INTERMEDIATE BETA VALUE
                beta_ramp = np.float32(beta*i/(pause_num-1))

                # COLLECT SYNAPTIC CURRENT AND FIRING RATES AT EACH BETA VALUE 
                dparams = net.weakly_clamped_phase(n_it_pos, epsilon, beta_ramp)
                dparams_history.append(dparams)

            for i in 1+np.arange(pause_num-1):

                # ASSUME NO HYSTERESIS AND USE INCREASING BETA VALUES FOR DECREASING BETA VALUES
                dparams_history.append(dparams_history[pause_num-1-i])

            # SET AMOUNT OF TIME FOR UP-RAMP (TOTAL CYCLE = 1 UNIT TIME)
            up_frac = .1

            # CONSTRUCT ARRAY OF TIMES FOR EACH BETA VALUE
            up_pulse_shape = up_frac*np.arange(pause_num)/(pause_num-1)
            down_pulse_shape = up_frac + (1-up_frac)*(1+np.arange((pause_num-1)))/(pause_num-1)
            pulse_shape = np.concatenate((up_pulse_shape,down_pulse_shape))

            # SET FORM OF MEMORY KERNEL (IMPORTANT TO HAVE EQUAL POSITIVE AND NEGATIVE LOBES I.E. np.sum(kernel) = 0)
            kernel = np.sin(np.arange(10)/10. * 2 * np.pi)/3
            
            # SET TIME RESOLUTION FOR INTERPOLATION GRID OF TOTAL CYCLE
            time_res = 100
            eval_grid = np.arange(time_res)/np.float(time_res)

            db_int = []
            dw_int = []

            # SET LAYER DEPENDENT NONLINEARITY THRESHOLDS
            cutoffs = [1e-5, 2e-4, 1.2e-6, 2e-5]

            # LOOP THROUGH EACH LAYER GETTING UPDATED INDIVIDUALLY
            for i in 1+np.arange(len(dparams_history[0])-1):

                single_param_history = np.array([h[i] for h in dparams_history])

                # INTERPOLATE SYNAPTIC CURRENTS/FIRING RATES ACROSS THE SAWTOOTH CYCLE
                single_param_func = interp1d(pulse_shape,single_param_history,axis=0)
                single_param_interp = single_param_func(eval_grid)

                # PERFORM KERNEL CONVOLUTION OF EACH SYNAPTIC CURRENT/FIRING RATE
                # ASSUME PERIODIC BOUNDARY CONDITIONS
                dpdt = convolve1d(single_param_interp, kernel, axis=0, mode='wrap') / time_res 

                # APPLY THRESHOLD NONLINEARITY
                dpdt[np.abs(dpdt) < cutoffs[i-1]] = 0

                # PERFORM INTEGRATION AND SCALE FOR WEIGHT UPDATE
                dp_int = .5*trapz(dpdt, eval_grid, axis=0) / beta * 10000

                if i == 1 or i == 2:
                    db_int.append(np.float32(dp_int))

                if i == 3 or i == 4:
                    dw_int.append(np.float32(dp_int))

            # UPDATE WEIGHTS
            func_input = db_int + dw_int + alphas
            Delta_logW = net.update_func( *func_input )
            gW = [gW1 + Delta_logW1 for gW1,Delta_logW1 in zip(gW,Delta_logW)]

        stdout.write("\n")
        dlogW = [100. * gW1 / n_batches_train for gW1 in gW]
        print "   "+" ".join(["dlogW%i=%.3f%%" % (k+1,dlogW1) for k,dlogW1 in enumerate(dlogW)])

        duration = (time.clock() - start_time) / 60.
        print("   duration=%.1f min" % (duration))

        ### VALIDATION ###
        
        # CUMULATIVE SUM OF VALIDATION ENERGY, VALIDATION COST AND VALIDATION ERROR
        measures_sum = [0.,0.,0.]

        for index in xrange(n_batches_valid):

            # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
            net.change_mini_batch_index(n_batches_train+index)

            # FREE PHASE
            net.free_phase(n_it_neg, epsilon)
            
            # MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
            measures = net.measure()
            measures_sum = [measure_sum + measure for measure_sum,measure in zip(measures_sum,measures)]
            measures_avg = [measure_sum / (index+1) for measure_sum in measures_sum]
            measures_avg[-1] *= 100. # measures_avg[-1] corresponds to the error rate, which we want in percentage
            stdout.write("\r   valid-%5i E=%.1f C=%.5f error=%.2f%%" % ((index+1)*batch_size, measures_avg[0], measures_avg[1], measures_avg[2]))
            stdout.flush()

            net.training_curves["validation error"].append(measures[-1]*100)

        stdout.write("\n")

        # SAVE THE PARAMETERS OF THE NETWORK AT THE END OF THE EPOCH
        net.save_params()


# HYPERPARAMETERS FOR A NETWORK WITH 1 HIDDEN LAYER
net1 = "net1", {
"hidden_sizes" : [500],
"n_epochs"     : 50,
"batch_size"   : 20,
"n_it_neg"     : 20,
"n_it_pos"     : 4,
"epsilon"      : np.float32(.5),
"beta"         : np.float32(.5),
"alphas"       : [np.float32(.1), np.float32(.05)]
}

if __name__ == "__main__":

    # TRAIN A NETWORK WITH 1 HIDDEN LAYER
    net = Network(*net1)
    train_net(net)
    np.save('val_error_convolution.npy', net.training_curves["validation error"])
    np.save('train_error_convolution.npy', net.training_curves["training error"])
