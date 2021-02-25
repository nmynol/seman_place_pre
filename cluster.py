import os

# for root, dirs, files in os.walk("./alldata/SinaDataset"):
#     # for file in files:
#     #     # 获取文件所属目录
#     #     print(root)
#     #     # 获取文件路径
#     #     print(os.path.join(root, file))
#     print(dirs)


def dp(root):
    file_list = os.listdir(root)
    print(file_list)
    for file in file_list:
        dp(os.path.join(root, file))


if __name__ == '__main__':
    # root = "./alldata"
    # root_list = os.listdir(root)
    # print(root_list)
    # for file in root_list:
    #     file_root = (os.path.join(root, file))
    #     if os.path.isdir(file_root):
    #         if os.path.exists(os.path.join(file_root, "alltext.txt")):
    #             fp = open(os.path.join(file_root, "alltext.txt"), 'r')
    #             for line in fp:
    #                 fq = open('./weibo.txt', 'a')  # 这里用追加模式
    #                 fq.write(line)
    #             fp.close()
    #             fq.close()
    root = "./alldata/test"
    root_list = os.listdir(root)
    for file in root_list:
        print(file)
        fp = open(os.path.join(root, str(file)), 'r')
        for line in fp:
            fq = open('./alldata/train/' + str(file), 'a')  # 这里用追加模式
            fq.write(line)
        fp.close()
        fq.close()
