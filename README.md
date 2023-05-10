# EjectionFractionPrediction
Ejection Fraction Prediction model without requiring segmentation, achieving MAE of 4.12% using labelled EchoNet-Dynamic data and unlabelled EchoNet-Pediatric data.

EjectionFractionPrediction and its dependencies can be installed by navigating to the cloned directory and running

    pip install --user .

Running the model

The model can be run by navigating to the cloned directory and running

    efpredict predict

You will need to request access to the EchoNet-Dynamic dataset by completing the form on this page: https://echonet.github.io/dynamic/index.html#access
You will need to request access to the EchoNet-Pediatric dataset by completing the form on this page: https://echonet.github.io/lvh/index.html#access
You will need to request access to the EchoNet-LVH (Not essential) dataset by completing the form on this page: https://echonet.github.io/pediatric/index.html#access

Once you have access to the data, download it and write the path of the EchoNet datasets in the efpredict.cfg file. 

### Some shuffling of data is required to run the model, namely in EchoNet-Pediatric.
