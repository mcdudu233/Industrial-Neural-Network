import pandas as pd


# 提取原始数据 去除没必要的参数
def transform_from_origin(key, file):
    data = pd.read_csv("./data/" + file)
    data = data.iloc[:, 1:]

    csv = pd.DataFrame(data=data[15:].values, columns=data[8:9].values.flatten())
    print(csv)
    csv.to_csv("./data/" + key + ".csv", index=False)


# 合并原始数据并打乱
def merge(files):
    csv = pd.DataFrame(
        columns=[
            "Channel1",
            "Channel2",
            "Channel3",
            "Channel4",
            "Channel5",
            "Channel6",
            "Channel7",
            "Channel8",
            "Type",
        ]
    )
    for key in files:
        data = pd.read_csv("./data/" + key + ".csv")
        data["Type"] = key
        csv = pd.concat([csv, data], ignore_index=True)
    print(csv.describe())

    # 打乱数据
    csv.sample()

    # 保存
    csv.to_csv("./data/data.csv", index=False)
    csv.describe().to_csv("./data/describe.csv")


# 分割为为 训练集 和 测试集
def split():
    # 训练集比例
    sector = 0.8

    csv = pd.read_csv("./data/data.csv")

    sp = int(csv.count().values[0] * sector)
    train = csv[:sp]
    test = csv[sp:]

    # 保存
    train.to_csv("./data/train.csv", index=False)
    test.to_csv("./data/test.csv", index=False)


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

    # 提取原始数据
    # for key in files:
    #     transform_from_origin(key, files[key])

    # merge(files)

    #      , Channel1, Channel2, Channel3, Channel4, Channel5, Channel6, Channel7, Channel8
    # count, 5242800.0, 5242800.0, 5242800.0, 5242800.0, 5242800.0, 5242800.0, 5242800.0, 5242800.0
    # mean, -0.13227930850976577, 0.0009032072751201647, 0.000668213336575875, 0.0010861044115739683, 8.736464503700307e-05, -0.00042250074082551245, 0.00033605961318379496, -0.00011272781509880225
    # std, 0.021324149408854822, 0.008439523316886572, 0.01693572440418254, 0.012510887614921454, 0.020240274870162618, 0.016078401353789913, 0.01790334344388813, 0.02377909824570553
    # min, -0.447809, -0.060058, -0.186553, -0.109949, -0.097331, -0.145065, -0.157163, -0.205172
    # 25 %, -0.142407, -0.004333, -0.009463, -0.006517, -0.014267, -0.010227, -0.01008, -0.013033
    # 50 %, -0.130934, 0.000957, 0.000664, 0.001079, -0.003138, -0.000446, 0.000477, -4.2e-05
    # 75 %, -0.119635, 0.006195, 0.010822, 0.008668, 0.012058, 0.009365, 0.010935, 0.012901
    # max, -0.049958, 0.072213, 0.166225, 0.114346, 0.110906, 0.152502, 0.163131, 0.216211

    split()
