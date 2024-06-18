
# Student GPA Prediction App

View our app here ➡️ [Student GPA Prediction App](https://kaggleperformancestudent-xxt98crdkxyv7awn8fduwk.streamlit.app/)

The Student GPA Prediction App is designed to predict the Grade Point Average (GPA) of a student based on several input factors such as weekly study time, number of absences, tutoring, parental support, and participation in extracurricular activities.

![alt text](logo.png)

## Purpose

The purpose of this project is to provide a tool for estimating a student's GPA, aiding in educational planning and student support initiatives.

## Getting Started

To run the application locally, follow these steps:

### Installation
Clone the repository and navigate into the project directory.
```
git clone https://github.com/iliyaskurmangali/kaggle_performance_student.git
cd iliyaskurmangali/kaggle_performance_student
```

### Installing Packages (Linux/WSL)
```
pip install -r requirements.txt
```
### Running Streamlit App locally (Linux/WSL)
```
streamlit run app/app.py
```

The app will open in your default web browser at http://localhost:8501

### Accuracy

The model has been fine-tuned with the specified parameters to achieve a high level of accuracy, as evidenced by a strong R² value of 0.9296, indicating that approximately 93% of the variance in the target variable is explained by the model. The errors are relatively low, with an MAE of 0.1897 and an RMSE of 0.2480, suggesting good predictive performance.

## Acknowledgements
- [Student Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset/data) for data

## License
This project is licensed under the MIT License
