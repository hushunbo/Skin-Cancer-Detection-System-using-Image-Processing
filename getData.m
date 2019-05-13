function getData()
%adds an extraction angle per pixel
offsets = [0 1; -1 1;-1 0;-1 -1;2 2];
jpgImagesDir = fullfile('Dataset/Test', '*.jpg');
total = numel( dir(jpgImagesDir) );
jpg_files = dir(jpgImagesDir);
jpg_counter = 0;
%total=length(filename);
gambar={total};
data_feat={total};
stats={total};
data_label={total};
label=1; 
limit=20; 

j=1;
for i=1:total
    
    %msgbox(jpg_files(j).name)
    s=strcat(num2str(i),'.jpg');
    file=fullfile('Dataset/Test',s);
    %file=fullfile(pathname,filename{i});
    gambar{i}=imread(file);
    gambar{i}=imresize(gambar{i},[600 600]);
    gambar{i}=rgb2gray(gambar{i});
%     gambar{i}=imadjust(gambar{i},stretchlim(gambar{i} ));
%     gambar{i}=imsharpen(gambar{i},'Radius',1,'Amount',0.5);
    glcm=graycomatrix(gambar{i}, 'Offset', offsets, 'Symmetric', true);
    stats{i}=graycoprops(glcm);
    


    iglcm=1;
    for x=1:5
      data_feat{i,x}=stats{i}.Contrast(iglcm);
      iglcm=iglcm+1;
    end
    iglcm=1;
    for x=6:10
        data_feat{i,x}=stats{i}.Correlation(iglcm);
        iglcm=iglcm+1;
    end
    iglcm=1;
    for x=12:16
        data_feat{i,x}=stats{i}.Energy(iglcm);
        iglcm=iglcm+1;
    end
        iglcm=1;
    for x=18:22
        data_feat{i,x}=stats{i}.Homogeneity(iglcm);
        iglcm=iglcm+1;
    end
        data_feat{i,24}=mean2(gambar{i});
        data_feat{i,25}=std2(gambar{i});
        data_feat{i,26}=entropy(gambar{i});
        data_feat{i,27}= mean2(var(double(gambar{i}))); %average image variance
        data_feat{i,28}=kurtosis(double(gambar{i}(:)));
        data_feat{i,29}=skewness(double(gambar{i}(:)));
        
        %labeling
        if i>limit
            label=label+1;
            data_label{i}=label;
            limit=limit+20;
        else
            data_label{1,i}=label;
        end         
end
% data is converted to the appropriate data type so that svm is not confused
data_feat1=cell2mat(data_feat);
disp(data_feat);
data_label=cell2mat(data_label);
save('Test_data.mat','data_feat1','data_label');