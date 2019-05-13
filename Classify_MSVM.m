function [Class_test] = Classify_MSVM(test_data,label,svmstruct,level)


[lg]=size(test_data);
for i=1:lg(1)
test=test_data(i,:);
cond=1;
indx=1;
while cond==1
    class_x=svmclassify(svmstruct{indx,1},test);
    if (ismember(class_x,label))
        Class_test(i)=class_x;
        cond=0;
    else
        %indx=find(level==class_x)+1;
        indx=isequal(level,class_x)+1;
        class_x=svmclassify(svmstruct{indx,1},test);
        if ismember(class_x,label);
        Class_test(i)=class_x;
        cond=0;
        else
        %indx=find(level==class_x)+1;
        indx=isequal(level,class_x)+1;
        end    
    end
   
end
end

