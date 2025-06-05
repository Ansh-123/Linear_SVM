import SVM

data_dim = 2
data_raw = [[-1, -1, -1], [1, 0, 1], [1, 2, 1], [2, 1, 1], [-1, -2, -1], [-2, -1, -1]]

margin = SVM.Line([1, -1], 0)
svm = SVM.Svm(margin, data_raw, data_dim)

svm.plot_data()
svm.train(0.01, 0.01, 0.01, 500)
svm.plot_data()

print(svm.margin.w)
