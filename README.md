# SpendWise AI — Customer Spending Prediction

A beginner-friendly machine-learning web application that classifies customers into **Low**, **Medium**, or **High** spending segments.

## Features
- Upload a customer CSV or use the included sample dataset.
- Data cleaning: standardizes column names, removes duplicate rows and handles missing values inside preprocessing pipelines.
- Exploratory analysis with more than five visual views.
- Feature engineering: creates a spending category from `SpendingScore` (or derives segments from a purchase amount column).
- Logistic Regression and Random Forest classification.
- Accuracy, weighted precision, weighted recall and weighted F1-score.
- Confusion matrix and Random Forest feature importance.
- Interactive prediction for a new customer.
- Business insights and cleaned CSV download.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn and Streamlit.

> Jupyter Notebook is not required. Run the complete project from the terminal.

## Quick start (Windows / VS Code)

1. Extract the ZIP file.
2. Open the extracted `customer_spending_prediction` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

```powershell
py -m venv .venv
```

5. Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

6. Install packages:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

7. Start the web app:

```powershell
py -m streamlit run app.py
```

8. Streamlit opens the app in your browser. If it does not, copy the `Local URL` shown in the terminal (usually `http://localhost:8501`) into Chrome.

To stop the server, click the terminal and press `Ctrl + C`.

## Dataset format

The included sample file is `data/sample_customers.csv`.

For your own CSV, include customer attributes such as age, income, gender, purchase amount, visits, or spending score. The app recognizes:
- An existing target: `SpendingCategory`, `SpendingLevel`, `SpendingClass`, or `Segment`.
- A `SpendingScore` column: creates Low (≤39), Medium (40–69), High (70+).
- A purchase amount column such as `PurchaseAmount`, `TotalSpend`, `SpendingAmount`, `AmountSpent`, or `AnnualSpend`: creates segments using lower, middle and upper thirds.

Avoid using customer names, phone numbers, email addresses or other personal identifiers in uploaded datasets. The app drops common ID columns from model training.

## GitHub publishing

Create a **new public repository** named `customer-spending-prediction` under `PranathiM15`. Do not reuse or overwrite `skill--lab-project`.

Repository description:
`AI-powered customer spending segmentation using Python, data analysis and machine learning.`

After creating the empty repository (do not add README, .gitignore or license on GitHub), run these commands from the project folder:

```powershell
git init
git add .
git status
git commit -m "Build AI customer spending prediction app"
git branch -M main
git remote add origin https://github.com/PranathiM15/customer-spending-prediction.git
git push -u origin main
```

If Git says `remote origin already exists`, inspect it:

```powershell
git remote -v
```

If it points to the wrong repository, replace it:

```powershell
git remote set-url origin https://github.com/PranathiM15/customer-spending-prediction.git
git push -u origin main
```

If Git is not installed, install Git for Windows from https://git-scm.com/download/win, restart VS Code, then check `git --version`.

## Important notes
- The sample dataset is synthetic and included for demonstration. Replace it with your assigned dataset if your mentor provided one.
- The spending-score thresholds are transparent project rules, not universal definitions of customer value.
- Model scores depend on dataset size, quality and the features supplied. Do not interpret high test accuracy as proof of real-world performance.
