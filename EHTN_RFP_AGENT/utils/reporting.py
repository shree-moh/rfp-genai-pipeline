import pandas as pd

def write_to_excel(data, filename="report.xlsx"):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f"Report saved as {filename}")
