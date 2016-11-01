(import [__future__ [print_function]]
        [keras.datasets [cifar10]]
        [keras.models [Sequential]]
        [keras.layers.core [Dense Dropout Activation Flatten]]
        [keras.layers.convolutional [Convolution2D MaxPooling2D]]
        [keras.optimizers [SGD Adadelta Adagrad]]
        [keras.utils [np_utils generic_utils]]
        [six.moves [range]]
        [numpy :as np])

(def batch_size 32
     nb_classes 10
     nb_epoch 100
  
     ;; input image dimensions
     img_rows 32
     img_cols 32
     
     ;; the CIFAR10 images are RGB
     img_channels 3)

(defn load-dataset []
  ;; the data, shuffled and split between train and test sets
  (let [[TRAIN TEST] (cifar10.load_data)
        [X-train y-train] TRAIN
        [X-test y-test] TEST

        ;; convert class vectors to binary class matrices
        Y-train (np_utils.to_categorical y-train nb_classes)
        Y-test (np_utils.to_categorical y-test nb_classes)]
    (print "OK")
    (print "X_train shape: " X-train.shape)
    (print (get X-train.shape 0) "train samples")
    (print (get X-test.shape 0) "test samples")

    ;; hy tuple notation
    (, X-train Y-train X-test Y-test)))

(defn make-network []
  (doto (Sequential)
        (.add (Convolution2D
               32 3 3
               :border_mode "same"
               :input_shape (, img_rows img_cols img_channels)))
        (.add (Activation "relu"))
        (.add (Convolution2D 32 3 3))
        (.add (Activation "relu"))
        (.add (MaxPooling2D :pool_size (, 2 2)))
        (.add (Dropout 0.25))

        (.add (Convolution2D 64 3 3 :border_mode "same"))
        (.add (Activation "relu"))
        (.add (Convolution2D 64 3 3))
        (.add (Activation "relu"))
        (.add (MaxPooling2D :pool_size (, 2 2)))
        (.add (Dropout 0.25))

        (.add (Flatten))
        (.add (Dense 512))
        (.add (Activation "relu"))
        (.add (Dropout 0.5))
        (.add (Dense nb_classes))
        (.add (Activation "softmax"))))

(defn train-model [model X-train Y-train X-test Y-test]
  (let [sgd (SGD :lr 0.01 :decay 1e-6 :momentum 0.9 :nesterov true)]
    (doto model
          (.compile :loss "categorical_crossentropy" :optimizer sgd)
          (.fit X-train Y-train
                :nb_epoch nb_epoch :batch_size batch_size
                :validation_split 0.1 :show_accuracy true
                :verbose 1))
    (print "Testing..."))
  (let [res (model.evaluate X-test Y-test
                            :batch_size batch_size
                            :verbose 1
                            :show_accuracy true)]
    (-> "Test accuracy {}"
        (.format (get res 1))
        (print))))

(defn save-model [model]
  (with [ofile (open "cifar10_architecture.json" "w")]
        (ofile.write (model.to_json)))
  (model.save_weights "cifar10_weights.h5"
                      :overwrite true))

(when (= --name-- "__main__")
  ;; for reproducibility
  (np.random.seed 1)
  
  (let [[X-train Y-train X-test Y-test] (load-dataset)]
    (doto (make-network)
          (train-model X-train Y-train X-test Y-test)
          (save-model))))
