
clear all;
clc;
cd 'D:\...';    % Dataset Folder
T = readtable('ConfLongDemo_JSI.csv');
H=height(T);
m=0;

for N=1:H       
    
   t=strcmp(T.activity(N,1),'onallfours');
   if t==0
       m=m+1;
       A(m,1) = T.Seq_name(N,1);
       B(m,1) = T.Tag_ID(N,1);
       C(m,1) = T.Time(N,1);
       D(m,1) = T.x(N,1);
       D(m,2) = T.y(N,1);
       D(m,3) = T.z(N,1);
       O(m,1) = T.activity(N,1);
   else disp(m); 
   end
       
end
clear N;
NewTab = table;
NewTab(:,1)=A(:,1);
X=array2table(B(:,1));
NewTab(:,2)=X;
NewTab(:,3)=C(:,1);

%Pre-Procesing
X1=hampel(D(:,1),4);                %Replacing outliers with hampel filter.
X2=hampel(D(:,2),4);
X3=hampel(D(:,3),4);
window=5;
X11=movmean(X1,window);             %Smoothing
X22=movmean(X2,window);
X33=movmean(X3,window);
NewTab(:,4)=array2table(X11);
NewTab(:,5)=array2table(X22);
NewTab(:,6)=array2table(X33);

%Feature Extraction
asd=D(:,1);
qwe=D(:,2);
zxc=D(:,3);
cde1=2*(asd-qwe);                       %Difference between 2 axes
cde2=2*(asd-zxc);
bnm1=array2table(cde1);
bnm2=array2table(cde2);
NewTab(:,7)=bnm1;
NewTab(:,8)=bnm2;
jkl=table2array(NewTab(:,[4 5 6]));
iop=2*rms(jkl,2);                     %RMS
xxx=array2table(iop);
NewTab(:,9)=xxx;
x11=asd-mean(asd); NewTab(:,10)=array2table(x11);
x12=qwe-mean(qwe); NewTab(:,11)=array2table(x12);
x13=zxc-mean(zxc); NewTab(:,12)=array2table(x13);
NewTab(:,13)=O(:,1);

clear X;clear X1;clear X2;clear X3;clear asd;clear qwe;clear zxc;clear cde1;clear cde2;clear bnm1;clear bnm2;
clear jkl;clear iop;clear xxx;clear x11;clear x12;clear x13;clear X11;clear X22;clear X33;

%Training
Inp = NewTab(1:86041,[2 4 5 6 7 8 9 10 11 12]);
Out = NewTab(1:86041,13);
X=table2array(Inp);
Y=table2array(Out);
classes = unique(Y);
M = fitcknn(Inp,Out,'NumNeighbors',5);          %kNN
for i=1:40                                      % Detecting cross validation error for different k values
    M = fitcknn(Inp,Out,'NumNeighbors',i);
    CV1 = crossval(M,'kFold',5);
    classError{i} = kfoldLoss(CV1);
end
for j = 1:numel(classes)                        %SVM
    indx = strcmp(Y,classes(j));                % Create binary classes for each classifier
    SVM{j} = fitcsvm(X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end
clear indx;
NB=fitcnb(X,Y);                                 %Naive Bayes
DT=fitctree(X,Y);                               %Decision Tree

%Cross-Validation
CV1 = crossval(M,'kFold',5);
classErrorknn = kfoldLoss(CV1)

for j = 1:numel(classes)
    CV2{j}=crossval(SVM{j},'kFold',5);
    classErrorsvm{j}=kfoldLoss(CV2{j})
end

CV3 = crossval(NB,'kFold',5);
classErrorNB = kfoldLoss(CV3)

CV4 = crossval(DT,'kFold',5);
classErrorDT = kfoldLoss(CV4)

%Testing(Prediction)
XP=table2array(NewTab(86042:159650,[2 4 5 6 7 8 9 10 11 12]));
YP=table2array(NewTab(86042:159650,13));
label1 = predict(M,XP);                  %kNN
c1=confusionmat(YP,label1);

for j = 1:numel(classes)                                %SVM
    label2 = predict(SVM{j},XP);
    Labels(:,j) = label2(:,1);
    indx(:,j) = strcmp(YP,classes(j));
    c2{j} = confusionmat(indx(:,j),Labels(:,j));
    
end

label3 = predict(NB,XP);
c3=confusionmat(YP,label3);

label4= predict(DT,XP);
c4=confusionmat(YP,label4);

%Prediction metrics
TP=0;
FP=0;
for i= 1:numel(classes)                     %kNN
TP=TP+c1(i,i);
for j= 1:numel(classes)
    if j~=i
        FP=FP+c1(i,j);
    end
end
end
Accuracy=2*TP/(2*TP+2*FP);                  %Since, True Positive(TP)=True Negetive(TN) and False Positive(FP)=False Negetive(FN)
Precision=TP/(TP+FP);
Fscore= 2*TP/(2*TP+2*FP);

for j= 1:numel(classes)             %SVM
    a= c2{j};
    TP2=a(2,2);
    FP2=a(1,2);
    TN2=a(1,1);
    FN2=a(2,1);
    Accuracy2{j}=(TP2+TN2)/(TP2+TN2+FP2+FN2);
    Precision2{j}=TP2/(TP2+FP2);
    Fscore2{j}=2*TP2/(2*TP2+FP2+FN2);
end

TP3=0;
FP3=0;
for i= 1:numel(classes)             %Naive Bayes
TP3=TP3+c3(i,i);
for j= 1:numel(classes)
    if j~=i
        FP3=FP3+c3(i,j);
    end
end
end
Accuracy3=2*TP3/(2*TP3+2*FP3);      %Since, True Positive(TP)=True Negetive(TN) and False Positive(FP)=False Negetive(FN)
Precision3=TP3/(TP3+FP3);
Fscore3= 2*TP3/(2*TP3+2*FP3);

TP4=0;
FP4=0;
for i= 1:numel(classes)             %Decision Tree
TP4=TP4+c4(i,i);
for j= 1:numel(classes)
    if j~=i
        FP4=FP4+c4(i,j);
    end
end
end
Accuracy4=2*TP4/(2*TP4+2*FP4);      %Since, True Positive(TP)=True Negetive(TN) and False Positive(FP)=False Negetive(FN)
Precision4=TP4/(TP4+FP4);
Fscore4= 2*TP4/(2*TP4+2*FP4);
