# California Housing Mini Project (APR Assignment-1)

This project applies **Linear Regression** and **Logistic Regression** on the California Housing Dataset (successor of California Housing, since California dataset is deprecated).  

The goal is to:
- Predict house prices using **Linear Regression**.
- Classify houses as **Cheap (0)** or **Expensive (1)** using **Logistic Regression**.

---

## 📂 Project Structure

Housing-Prediction-APR-Project/
│
├── src/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│
├── outputs/
├── requirement.txt
├── README.md
└── report.pdf

---

## ⚙️ Requirements
Make sure you have **Python 3.8+** installed.  
Install dependencies with:

```bash
pip install -r requirements.txt
```

Contents of requirements.txt:
numpy
pandas
matplotlib
seaborn
scikit-learn

-------------------------------------------
How to Run

1. Activate virtual environment (if using one):

# Windows:    
    venv\Scripts\activate
# Mac/Linux:  
    source venv/bin/activate

2. Run Linear Regression:
       python src/linear_regression.py

3. Run Logistic Regression:
      python src/logistic_regression.py

--------------------------------------------

How to push on GitHub 

1.Using the Git Command Line (for larger projects and folders):
  Initialize Git: Open your terminal or command prompt, navigate to the root directory of your project folder, and initialize a Git repository:
   
    git init
2.Add files to staging: Add all files and folders within your project to the staging area:

    git add .
3.Commit changes: Commit the staged changes with a descriptive message:

    git commit -m "Initial commit of my project folder"
4.Connect to GitHub repository: Link your local repository to your GitHub repository. Replace yourUsername and yourRepository with your actual GitHub username and repository name:

    git remote add origin [https://github.com/yourUsername/yourRepository.git](https://github.com/NITINSINGH321/APR_minipro.git)
5.Push to GitHub: Push your local changes to the main (or master) branch of your GitHub repository:

    git push -u origin main
