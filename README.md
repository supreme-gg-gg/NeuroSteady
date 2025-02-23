# Hand Tremor Intervention in Neurosurgery

The project uses few-shot transfer learning with CNN-LSTM to detect hand tremors of neurosurgeons during surgery, ensembled with a simple majority voting system with a sliding window time-frequency analysis. This triggers an intervention using a wearable device involving servo motors driving mechanical exoskeletons to stabilize the hand tremors when operating on patients.

> This project is submitted to BMEC NeuroHacks 2025 at the University of Toronto.

> View our demo... (DNE for now)

## Background

To be written

## Dataset

We trained the base model on the [Hand Tremor Dataset](https://www.kaggle.com/datasets/aaryapandya/hand-tremor-dataset-collected-using-mpu9250-sensor) collected using MPU9250 sensor. This is because Parkinson's tremor dataset is widely available but our use case with smaller tremors is not well represented in the literature.

We then obtained our custom dataset by simulating tremors and collecting them using a similar MPU6050 sensor on Arduino (code available in `data_collection`). This very small dataset is used for few-shot learning.

## Model

The model is a CNN-LSTM architecture with a few-shot learning approach. 1D CNN is used to extract features from the time series data and LSTM is used to capture the temporal dependencies. The model is trained on the base dataset and fine-tuned on the custom dataset.

The base model achieved an accuracy of 80% with minor optimization and little overfitting.

During fine tuning, we freeze the CNN-LSTM layers and only retrain the feed forward layers. This is because the base dataset is large and the custom dataset is small. The final accuracy obtained on our own dataset in 78%.

This is then ensembled with a simple majority voting system with a sliding window time-frequency analysis. The model is run on the windowed data and the majority vote is taken. This is done to reduce the false positives and false negatives, given that the CNN-LSTM model is not perfect and quite sensitive to the input data, while traditional signal processing methods are not sensitive enough but are robust.

## Intervention

The processing is done on the computer due to limited hardware, but can be easily transferred to a Raspberry Pi or similar device. The intervention is triggered when the majority vote is above a certain threshold and a signal is sent via Serial connection to Arduino UNO. The intervention involves a wearable device with servo motors driving mechanical exoskeletons to stabilize the hand tremors when operating on patients.

To be written...
