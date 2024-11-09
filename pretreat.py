import pandas as pd


# 提取原始数据 去除没必要的参数
def transform_from_origin(key, file):
    data = pd.read_csv("./data/" + file)
    data = data.iloc[:, 1:]

    csv = pd.DataFrame(data=data[15:].values, columns=data[8:9].values.flatten())
    print(csv)
    csv.to_csv("./data/" + key + ".csv", index=False)


if __name__ == "__main__":
    # 总共有4种类型的错误和健康状况
    kinds = pd.read_csv("./data/4-kinds-of-gear-faults")
    print(kinds)

    # 每种状态所对应的文件
    files = {
        "Chipped": "Chipped_30_2.csv",
        "Missing": "Miss_30_2.csv",
        "Health": "Health_30_2.csv",
        "Root": "Root_30_2.csv",
        "Surface": "Surface_30_2.csv",
    }

    for key in files:
        transform_from_origin(key, files[key])
