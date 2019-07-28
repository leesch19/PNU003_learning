import main

model = main.main()

model.training(epo=50000, continue_training=False, l2_norm=True, first_training=True)

#model.keep_prob = 1.0
#model.evaluation()