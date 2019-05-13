load('data_1.mat');
load('Test_data.mat');
result1 = svm(data_feat,data_label,data_feat1);
%[Cmat,Accuracy]= confusion_matrix(data_label,result1,{'Desert','Forest','Mountain','Residential','River'});
[c_matrixp,Result]= confusion.getMatrix(data_label,result1);
%ConfusionMatrix = confusionmat(data_label,result1)
%stats = confusionmatStats(data_label,result1);

