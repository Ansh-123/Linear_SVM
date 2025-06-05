import SVM

data_dim = 2
data_raw = [[-1, -1, -1], [1, 0, 1], [1, 2, 1], [2, 1, 1], [-1, -2, -1], [-2, -1, -1]]
data_processed = []

margin = SVM.Line([1, -1], 0)
svm = SVM.Svm(margin, data_raw, data_dim)

svm.plot_data()
svm.train(0.01, 0.01, 0.01, 500)
svm.plot_data()

print(svm.margin.w)

support_vectors = svm.find_support_vectors()
print(svm.calculate_loss())
print(svm.calculate_grad_weights())
