import joblib


tmp = {"mean": 10, "var": 20}

joblib.dump(tmp, "tmp.pkl")

print(joblib.load("tmp.pkl"))
