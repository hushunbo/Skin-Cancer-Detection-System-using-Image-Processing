function varargout = main(varargin)


gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @main_OpeningFcn, ...
                   'gui_OutputFcn',  @main_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before main is made visible.
function main_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to main (see VARARGIN)

% Choose default command line output for main


% Update handles structure
guidata(hObject, handles);

axes(handles.axes1); axis off
% UIWAIT makes main wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = main_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
%varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname, filterindex]=uigetfile( ...
    {'*.jpg','JPEG File (*.jpg)'; ...
     '*.*','Any Image file (*.*)'}, ...
     'Pick an image file');
var=strcat(pathname,filename);
       k=imread(var);
       handles.YY = k;
       set(handles.edit1,'String',var);
       
z=k;
guidata(hObject,handles);
%guidata(hObject,handles);
axes(handles.axes1);
   imshow(k);
%   set(handles.axes1);
   title('Input Image');
      
set(handles.pushbutton1,'enable','off');    
set(handles.pushbutton16,'enable','on');


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
k=handles.YY;
j=rgb2gray(k);
handles.XX=j;
figure,subplot(2,2,2),imshow(j);title('gray Image');
set(handles.pushbutton1,'enable','off');    
set(handles.pushbutton2,'enable','off');
set(handles.pushbutton4,'enable','on');

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

str=get(handles.edit1,'String');
I = imread(str);
h = ones(5,5)/25;
I2 = imfilter(I,h);
figure
imshow(I2)
title('Filtered Image')
set(handles.pushbutton3,'enable','off');
set(handles.pushbutton12,'enable','on');


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ei=25;
st=35;
%k=10
k=ei*st;
I=handles.YY;
%I = imread('1.jpg');
%h=filter matrx
h = ones(ei,st) / k;
I1 = imfilter(I,h,'symmetric');
IG=rgb2gray(I1);
%Converting to BW
I11 = imadjust(IG,stretchlim(IG),[]);
level = graythresh(I11);
BWJ = im2bw(I11,level);
dim = size(BWJ)
IN=ones(dim(1),dim(2));
BW=xor(BWJ,IN);  %inverting
figure,subplot(2,2,2), imshow(BW), title('Black and White');
set(handles.pushbutton1,'enable','off');    
set(handles.pushbutton2,'enable','off');
set(handles.pushbutton3,'enable','on');
set(handles.pushbutton4,'enable','off');


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)





% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%I=imread('cancer.bmp');
I=handles.YY;
[y,x,z]=size(I);
myI=double(I);                 
H=zeros(y,x);
S=H;
HS_I=H;
for i=1:x
    for j=1:y
        HS_I(j,i)=((myI(j,i,1)+myI(j,i,2)+myI(j,i,3))/3);
        S(j,i)=1-3*min(myI(j,i,:))/(myI(j,i,1)+myI(j,i,2)+myI(j,i,3));
        if ((myI(j,i,1)==myI(j,i,2))&(myI(j,i,2)==myI(j,i,3)))       
            Hdegree=0;
        else    
            Hdegree=acos(0.5*(2*myI(j,i,1)-myI(j,i,2)-myI(j,i,3))/((myI(j,i,1)-myI(j,i,2))^2+(myI(j,i,1)-myI(j,i,3))*(myI(j,i,2)-myI(j,i,3)))^0.5);
        end    
        if (myI(j,i,2)>=myI(j,i,3))
            H(j,i)=Hdegree;                                    
        else
            H(j,i)=(2*pi-Hdegree);                              
        end     
    end 
end



Hth1=0.9*2*pi; Hth2=0.1*2*pi; 
Nred=0;                       
for i=1:x
    for j=1:y
        if ((H(j,i)>=Hth1)||(H(j,i)<=Hth2))
            Nred=Nred+1;       
        end
    end
end

Ratio=Nred/(x*y);           

if (Ratio>=0.6)              
    Red=1
else
    Red=0
end    


HS_I=uint8(HS_I);                                                    

figure(1);
imshow(I);
figure(2);
imshow(HS_I);


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
createffnn


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% NET = trainnet(net,IMGDB);
str=get(handles.edit1,'String');
I1 = imread(str);
I = im2double(I1);
HSV = rgb2hsv(I);
H = HSV(:,:,1); H = H(:);
S = HSV(:,:,2); S = S(:);
V = HSV(:,:,3); V = V(:);
idx = kmeans([H S V], 4);
imshow(I1);
figure,imshow(ind2rgb(reshape(idx, size(I,1), size(I, 2)), [0 0 1; 0 0.8 0]))

% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function [ Unow, center, now_obj_fcn ] = FCMforImage( img, clusterNum )


if nargin < 2
    clusterNum = 2;   % number of cluster
end

[row, col] = size(img);
expoNum = 2;      % fuzzification parameter
epsilon = 0.001;  % stopping condition
mat_iter = 100;   % number of maximun iteration


Upre = rand(row, col, clusterNum);
dep_sum = sum(Upre, 3);
dep_sum = repmat(dep_sum, [1,1, clusterNum]);
Upre = Upre./dep_sum;

center = zeros(clusterNum,1); 

for i=1:clusterNum
    center(i,1) = sum(sum(Upre(:,:,i).*img))/sum(sum(Upre(:,:,i)));
end

pre_obj_fcn = 0;
for i=1:clusterNum
    pre_obj_fcn = pre_obj_fcn + sum(sum((Upre(:,:,i) .*img - center(i)).^2));
end
%fprintf('Initial objective fcn = %f\n', pre_obj_fcn);

for iter = 1:mat_iter    

    Unow = zeros(size(Upre));
    for i=1:row
        for j=1:col

            for uII = 1:clusterNum
                tmp = 0;
                for uJJ = 1:clusterNum
                    disUp = abs(img(i,j) - center(uII));
                    disDn = abs(img(i,j) - center(uJJ));
                    tmp = tmp + (disUp/disDn).^(2/(expoNum-1));
                end
                Unow(i,j, uII) = 1/(tmp);
            end            
        end
    end   

    now_obj_fcn = 0;
    for i=1:clusterNum
        now_obj_fcn = now_obj_fcn + sum(sum((Unow(:,:,i) .*img - center(i)).^2));
    end
   % fprintf('Iter = %d, Objective = %f\n', iter, now_obj_fcn);

    if max(max(max(abs(Unow-Upre))))<epsilon || abs(now_obj_fcn - pre_obj_fcn)<epsilon 
        break;
    else
        Upre = Unow.^expoNum;
       
        for i=1:clusterNum
            center(i,1) = sum(sum(Upre(:,:,i).*img))/sum(sum(Upre(:,:,i)));
        end
        pre_obj_fcn = now_obj_fcn;
    end
end

% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton12.
function pushbutton12_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str=get(handles.edit1,'String');
I1 = imread(str);
str=rgb2gray(I1);
imwrite(str,'img.tif','tiff');
img = double(imread('img.tif'));
clusterNum = 2;
[ Unow, center, now_obj_fcn ] = FCMforImage( img, clusterNum );
figure;
subplot(2,2,1); imshow(img,[]);
for i=1:clusterNum
    subplot(2,2,i+1);
    imshow(Unow(:,:,i),[]);
    imwrite(Unow(:,:,i),'seg.jpg');
end
imshow('seg.jpg'); title('Segmented Image');

set(handles.pushbutton13,'enable','on');
set(handles.pushbutton12,'enable','off');

% --- Executes on button press in pushbutton13.
function pushbutton13_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
s=get(handles.edit1,'String');
I = imread(s);
I=rgb2gray(I);
glcms = graycomatrix(I);
getData()
% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
seg_img=imread('seg.jpg');
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
%Skewness = skewness(img)
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    
feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
img = imread(s);
 gaborArray = gaborFilterBank(5,8,39,39);  % Generates the Gabor filter bank
 featureVector = gaborFeatures(img,gaborArray,4,4);
 disp('Gabor Feature');
 featureVector
 save('Gaborfeature.mat','featureVector');
% GLCM2 = graycomatrix(I,'Offset',[2 0;0 2]);
% features = GLCM_Features1(GLCM2,0)
set(handles.pushbutton14,'enable','on');
set(handles.pushbutton13,'enable','off');
msgbox('Feature Extracted Done');
clear all;
function [out] = GLCM_Features1(glcmin,pairs)
if ((nargin > 2) || (nargin == 0))
   error('Too many or too few input arguments. Enter GLCM and pairs.');
elseif ( (nargin == 2) ) 
    if ((size(glcmin,1) <= 1) || (size(glcmin,2) <= 1))
       error('The GLCM should be a 2-D or 3-D matrix.');
    elseif ( size(glcmin,1) ~= size(glcmin,2) )
        error('Each GLCM should be square with NumLevels rows and NumLevels cols');
    end    
elseif (nargin == 1) 
    pairs = 0; 
    if ((size(glcmin,1) <= 1) || (size(glcmin,2) <= 1))
       error('The GLCM should be a 2-D or 3-D matrix.');
    elseif ( size(glcmin,1) ~= size(glcmin,2) )
       error('Each GLCM should be square with NumLevels rows and NumLevels cols');
    end    
end


format long e
if (pairs == 1)
    newn = 1;
    for nglcm = 1:2:size(glcmin,3)
        glcm(:,:,newn)  = glcmin(:,:,nglcm) + glcmin(:,:,nglcm+1);
        newn = newn + 1;
    end
elseif (pairs == 0)
    glcm = glcmin;
end

size_glcm_1 = size(glcm,1);
size_glcm_2 = size(glcm,2);
size_glcm_3 = size(glcm,3);


out.autoc = zeros(1,size_glcm_3); 
out.contr = zeros(1,size_glcm_3); 
out.corrm = zeros(1,size_glcm_3);
out.corrp = zeros(1,size_glcm_3); 
out.cprom = zeros(1,size_glcm_3); 
out.cshad = zeros(1,size_glcm_3); 
out.dissi = zeros(1,size_glcm_3);
out.energ = zeros(1,size_glcm_3); 
out.entro = zeros(1,size_glcm_3); 
out.homom = zeros(1,size_glcm_3); 
out.homop = zeros(1,size_glcm_3); 
out.maxpr = zeros(1,size_glcm_3);

out.sosvh = zeros(1,size_glcm_3); 
out.savgh = zeros(1,size_glcm_3);
out.svarh = zeros(1,size_glcm_3); 
out.senth = zeros(1,size_glcm_3); 
out.dvarh = zeros(1,size_glcm_3);

out.denth = zeros(1,size_glcm_3);
out.inf1h = zeros(1,size_glcm_3); 
out.inf2h = zeros(1,size_glcm_3); 

out.indnc = zeros(1,size_glcm_3);
out.idmnc = zeros(1,size_glcm_3); 



glcm_sum  = zeros(size_glcm_3,1);
glcm_mean = zeros(size_glcm_3,1);
glcm_var  = zeros(size_glcm_3,1);


u_x = zeros(size_glcm_3,1);
u_y = zeros(size_glcm_3,1);
s_x = zeros(size_glcm_3,1);
s_y = zeros(size_glcm_3,1);




p_x = zeros(size_glcm_1,size_glcm_3); 
p_y = zeros(size_glcm_2,size_glcm_3); 
p_xplusy = zeros((size_glcm_1*2 - 1),size_glcm_3); 
p_xminusy = zeros((size_glcm_1),size_glcm_3);

hxy  = zeros(size_glcm_3,1);
hxy1 = zeros(size_glcm_3,1);
hx   = zeros(size_glcm_3,1);
hy   = zeros(size_glcm_3,1);
hxy2 = zeros(size_glcm_3,1);



for k = 1:size_glcm_3 

    glcm_sum(k) = sum(sum(glcm(:,:,k)));
    glcm(:,:,k) = glcm(:,:,k)./glcm_sum(k); 
    glcm_mean(k) = mean2(glcm(:,:,k)); 
    glcm_var(k)  = (std2(glcm(:,:,k)))^2;
    
    for i = 1:size_glcm_1

        for j = 1:size_glcm_2

            out.contr(k) = out.contr(k) + (abs(i - j))^2.*glcm(i,j,k);
            out.dissi(k) = out.dissi(k) + (abs(i - j)*glcm(i,j,k));
            out.energ(k) = out.energ(k) + (glcm(i,j,k).^2);
            out.entro(k) = out.entro(k) - (glcm(i,j,k)*log(glcm(i,j,k) + eps));
            out.homom(k) = out.homom(k) + (glcm(i,j,k)/( 1 + abs(i-j) ));
            out.homop(k) = out.homop(k) + (glcm(i,j,k)/( 1 + (i - j)^2));
           
            out.sosvh(k) = out.sosvh(k) + glcm(i,j,k)*((i - glcm_mean(k))^2);
            
            
            out.indnc(k) = out.indnc(k) + (glcm(i,j,k)/( 1 + (abs(i-j)/size_glcm_1) ));
            out.idmnc(k) = out.idmnc(k) + (glcm(i,j,k)/( 1 + ((i - j)/size_glcm_1)^2));
            u_x(k)          = u_x(k) + (i)*glcm(i,j,k); 
            u_y(k)          = u_y(k) + (j)*glcm(i,j,k); 

        end
        
    end
    out.maxpr(k) = max(max(glcm(:,:,k)));
end


for k = 1:size_glcm_3
    
    for i = 1:size_glcm_1
        
        for j = 1:size_glcm_2
            p_x(i,k) = p_x(i,k) + glcm(i,j,k); 
            p_y(i,k) = p_y(i,k) + glcm(j,i,k); % taking i for j and j for i
            if (ismember((i + j),[2:2*size_glcm_1])) 
                p_xplusy((i+j)-1,k) = p_xplusy((i+j)-1,k) + glcm(i,j,k);
            end
            if (ismember(abs(i-j),[0:(size_glcm_1-1)])) 
                p_xminusy((abs(i-j))+1,k) = p_xminusy((abs(i-j))+1,k) +...
                    glcm(i,j,k);
            end
        end
    end
    

    
end


for k = 1:(size_glcm_3)
    
    for i = 1:(2*(size_glcm_1)-1)
        out.savgh(k) = out.savgh(k) + (i+1)*p_xplusy(i,k);

        out.senth(k) = out.senth(k) - (p_xplusy(i,k)*log(p_xplusy(i,k) + eps));
    end

end

for k = 1:(size_glcm_3)
    
    for i = 1:(2*(size_glcm_1)-1)
        out.svarh(k) = out.svarh(k) + (((i+1) - out.senth(k))^2)*p_xplusy(i,k);

    end

end

for k = 1:size_glcm_3

    for i = 0:(size_glcm_1-1)
        out.denth(k) = out.denth(k) - (p_xminusy(i+1,k)*log(p_xminusy(i+1,k) + eps));
        out.dvarh(k) = out.dvarh(k) + (i^2)*p_xminusy(i+1,k);
    end
end


for k = 1:size_glcm_3
    hxy(k) = out.entro(k);
    for i = 1:size_glcm_1
        
        for j = 1:size_glcm_2
            hxy1(k) = hxy1(k) - (glcm(i,j,k)*log(p_x(i,k)*p_y(j,k) + eps));
            hxy2(k) = hxy2(k) - (p_x(i,k)*p_y(j,k)*log(p_x(i,k)*p_y(j,k) + eps));

        end
        hx(k) = hx(k) - (p_x(i,k)*log(p_x(i,k) + eps));
        hy(k) = hy(k) - (p_y(i,k)*log(p_y(i,k) + eps));
    end
    out.inf1h(k) = ( hxy(k) - hxy1(k) ) / ( max([hx(k),hy(k)]) );
    out.inf2h(k) = ( 1 - exp( -2*( hxy2(k) - hxy(k) ) ) )^0.5;

end

corm = zeros(size_glcm_3,1);
corp = zeros(size_glcm_3,1);

for k = 1:size_glcm_3
    for i = 1:size_glcm_1
        for j = 1:size_glcm_2
            s_x(k)  = s_x(k)  + (((i) - u_x(k))^2)*glcm(i,j,k);
            s_y(k)  = s_y(k)  + (((j) - u_y(k))^2)*glcm(i,j,k);
            corp(k) = corp(k) + ((i)*(j)*glcm(i,j,k));
            corm(k) = corm(k) + (((i) - u_x(k))*((j) - u_y(k))*glcm(i,j,k));
            out.cprom(k) = out.cprom(k) + (((i + j - u_x(k) - u_y(k))^4)*...
                glcm(i,j,k));
            out.cshad(k) = out.cshad(k) + (((i + j - u_x(k) - u_y(k))^3)*...
                glcm(i,j,k));
        end
    end
    
    s_x(k) = s_x(k) ^ 0.5;
    s_y(k) = s_y(k) ^ 0.5;
    out.autoc(k) = corp(k);
    out.corrp(k) = (corp(k) - u_x(k)*u_y(k))/(s_x(k)*s_y(k));
    out.corrm(k) = corm(k) / (s_x(k)*s_y(k));

end
function gaborArray = gaborFilterBank(u,v,m,n)





if (nargin ~= 4)    % Check correct number of arguments
    error('There must be four input arguments (Number of scales and orientations and the 2-D size of the filter)!')
end


%% Create Gabor filters
% Create u*v gabor filters each being an m by n matrix

gaborArray = cell(u,v);
fmax = 0.25;
gama = sqrt(2);
eta = sqrt(2);

for i = 1:u
    
    fu = fmax/((sqrt(2))^(i-1));
    alpha = fu/gama;
    beta = fu/eta;
    
    for j = 1:v
        tetav = ((j-1)/v)*pi;
        gFilter = zeros(m,n);
        
        for x = 1:m
            for y = 1:n
                xprime = (x-((m+1)/2))*cos(tetav)+(y-((n+1)/2))*sin(tetav);
                yprime = -(x-((m+1)/2))*sin(tetav)+(y-((n+1)/2))*cos(tetav);
                gFilter(x,y) = (fu^2/(pi*gama*eta))*exp(-((alpha^2)*(xprime^2)+(beta^2)*(yprime^2)))*exp(1i*2*pi*fu*xprime);
            end
        end
        gaborArray{i,j} = gFilter;
        
    end
end


%% Show Gabor filters (Please comment this section if not needed!)

% Show magnitudes of Gabor filters:
figure('NumberTitle','Off','Name','Magnitudes of Gabor filters');
for i = 1:u
    for j = 1:v        
        subplot(u,v,(i-1)*v+j);        
        imshow(abs(gaborArray{i,j}),[]);
    end
end

% Show real parts of Gabor filters:
figure('NumberTitle','Off','Name','Real parts of Gabor filters');
for i = 1:u
    for j = 1:v        
        subplot(u,v,(i-1)*v+j);        
        imshow(real(gaborArray{i,j}),[]);
    end
end

function featureVector = gaborFeatures(img,gaborArray,d1,d2)



if (nargin ~= 4)        
    error('Please use the correct number of input arguments!')
end

if size(img,3) == 3     
    warning('The input RGB image is converted to grayscale!')
    img = rgb2gray(img);
end

img = double(img);



[u,v] = size(gaborArray);
gaborResult = cell(u,v);
for i = 1:u
    for j = 1:v
        gaborResult{i,j} = imfilter(img, gaborArray{i,j});
    end
end



featureVector = [];
for i = 1:u
    for j = 1:v
        
        gaborAbs = abs(gaborResult{i,j});
        gaborAbs = downsample(gaborAbs,d1);
        gaborAbs = downsample(gaborAbs.',d2);
        gaborAbs = gaborAbs(:);
        
        
        gaborAbs = (gaborAbs-mean(gaborAbs))/std(gaborAbs,1);
        
        featureVector =  [featureVector; gaborAbs];
        
    end
end

function getData()
%adds an extraction angle per pixel
offsets = [0 1; -1 1;-1 0;-1 -1;2 2];
jpgImagesDir = fullfile('Dataset/Train', '*.jpg');
total = numel( dir(jpgImagesDir) );
jpg_files = dir(jpgImagesDir);
jpg_counter = 0;
%total=length(filename);
gambar={total};
data_feat={total};
stats={total};
data_label={total};
label=1; 
limit=5; 

j=1;
for i=1:total
    
    %msgbox(jpg_files(j).name)
    s=strcat(num2str(i),'.jpg');
    file=fullfile('Dataset/Train',s);
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
            limit=limit+5;
        else
            data_label{1,i}=label;
        end         
end
% data is converted to the appropriate data type so that svm is not confused
data_feat=cell2mat(data_feat);
disp(data_feat);
data_label=cell2mat(data_label);
save('data_1.mat','data_feat','data_label');
% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
s=get(handles.edit1,'String');
test=imread(s);

test=imresize(test,[600 600]);
test=rgb2gray(test);
offsets = [0 1; -1 1;-1 0;-1 -1;2 2];

  glcm=graycomatrix(test, 'Offset', offsets, 'Symmetric', true);

stats=graycoprops(glcm);
data_glcm=struct2array(stats);
iglcm=1;
glcm_contrast={5};
glcm_correlation={5};
glcm_energy={5};
glcm_homogeneity={5};
    for x=1:5
      glcm_contrast{x}=data_glcm(iglcm);
      iglcm=iglcm+1;
    end
    for x=1:5
        glcm_correlation{x}=data_glcm(iglcm);
        iglcm=iglcm+1;
    end
    for x=1:5
        glcm_energy{x}=data_glcm(iglcm);
        iglcm=iglcm+1;
    end
    for x=1:5
        glcm_homogeneity{x}=data_glcm(iglcm);
        iglcm=iglcm+1;
    end
rata2=mean2(test);
std_deviation=std2(test);
glcm_entropy=entropy(test);
rata2_variance= mean2(var(double(test)));
glcm_kurtosis=kurtosis(double(test(:)));
glcm_skewness=skewness(double(test(:)));
buat_train=[glcm_contrast(1:5),glcm_correlation(1:5),glcm_energy(1:5),glcm_homogeneity(1:5),rata2,std_deviation,glcm_entropy,rata2_variance,glcm_kurtosis,glcm_skewness];

test_data=cell2mat(buat_train);

disp('GLCM Feature')
disp('Contrast(1) Correlation(2) Energy(3)  Homogeneity(4) Mean(5)  Standard_Deviation(6) Entropy(7) RMS(8) Variance(9) smoothness(10) Kurtosis(11) Skewness(12) IDM(13)')

input_Feature=test_data 

load('data_1.mat');
%load('Test_data.mat');
%disp('Gabor Feature');
load('Gaborfeature.mat');
%Input_Feat=test_data

result = multisvm(data_feat,data_label,test_data);

%result1 = multisvm(data_feat,data_label,data_feat1);
%[Cmat,Accuracy]= confusion_matrix(result1,data_label,{'Desert','Forest','Mountain','Residential','River'});
%[c_matrixp,Result]= confusion.getMatrix(data_label,result1);
%C = confusionmat(data_label,result1)
%disp(result);
 
if result == 1
    A1 = 'actinic keratosis';
    set(handles.edit2,'string',A1);
    helpdlg('actinic keratosis');
    disp('actinic keratosis');
elseif result == 2
    A2 = 'Basel cell carcinoma';
    set(handles.edit2,'string',A2);
    helpdlg('Basel cell carcinoma');
    disp('Basel cell carcinoma');
elseif result == 3
    A3 = 'cherry nevus';
    set(handles.edit2,'string',A3);
    helpdlg('cherry nevus');
    disp('cherry nevus');
elseif result == 4
    A4 = 'dermatofibroma';
    set(handles.edit2,'string',A4);
    helpdlg('dermatofibroma');
    disp('dermatofibroma');
elseif result == 5
    A5 = 'Melanocytic nevus';
    set(handles.edit2,'string',A5);
    helpdlg('Melanocytic nevus');
    disp('Melanocytic nevus');
    elseif result == 6
    A5 = 'Melanoma';
    set(handles.edit2,'string',A5);
    helpdlg('Melanoma');
    disp('Melanoma');
end

% function [itrfin] = multisvm( T,C,test )
% 
% 
% itrind=size(test,1);
% itrfin=[];
% Cb=C;
% Tb=T;
% itr=1;
% 
% for tempind=1:itrind
%     tst=test(tempind,:);
%     C=Cb;
%     T=Tb;
%     u=unique(C);
%     N=length(u);
%     c4=[];
%     c3=[];
%     j=1;
%     k=1;
%     if(N>2)
%         itr=1;
%         classes=0;
%         cond=max(C)-min(C);
%         while((classes~=1)&&(itr<=length(u))&& size(C,2)>1 && cond>0)
%         %This while loop is the multiclass SVM Trick
%             c1=(C==u(itr));
%             newClass=c1;
%             svmStruct = svmtrain(T,newClass,'kernel_function','rbf');   % I am using rbf kernel function, you must change it also
%             classes = svmclassify(svmStruct,tst);
%         
%             % This is the loop for Reduction of Training Set
%             for i=1:size(newClass,2)
%                 if newClass(1,i)==0;
%                     c3(k,:)=T(i,:);
%                     k=k+1;
%                 end
%             end
%         T=c3;
%         c3=[];
%         k=1;
%         
%             % This is the loop for reduction of group
%             for i=1:size(newClass,2)
%                 if newClass(1,i)==0;
%                     c4(1,j)=C(1,i);
%                     j=j+1;
%                 end
%             end
%         C=c4;
%         c4=[];
%         j=1;
%         
%         cond=max(C)-min(C); % Condition for avoiding group 
%                             %to contain similar type of values 
%                             %and the reduce them to process
%         
%             % This condition can select the particular value of iteration
%             % base on classes
%             if classes~=1
%                 itr=itr+1;
%             end    
%         end
%     end
% 
% valt=Cb==u(itr);		% This logic is used to allow classification
% val=Cb(valt==1);		% of multiple rows testing matrix
% val=unique(val);
% itrfin(tempind,:)=val;  
% end



% Give more suggestions for improving the program.
function [result] = multisvm(TrainingSet,GroupTrain,TestSet)


u=unique(GroupTrain);
numClasses=length(u);
result = zeros(length(TestSet(:,1)),1);

%build models
for k=1:numClasses
    %Vectorized statement that binarizes Group
    %where 1 is the current class and 0 is all other classes
    G1vAll=(GroupTrain==u(k));
    models(k) = svmtrain(TrainingSet,G1vAll);
end

%classify test cases
for j=1:size(TestSet,1)
    for k=1:numClasses
     
        if(svmclassify(models(k),TestSet(j,:))) 
            break;
        end
    end
    result(j) = k;
    %[Cmat,Accuracy]= confusion_matrix(TestSet(j,:),models(k),{'A','B','C','D','E','F'});
    %disp('Result')
    %disp(result(j));
    
end
function exCode1()
folder = 'Dataset'; 
dirImage = dir( folder ); 

numData = size(dirImage,1); 

M ={} ; 




for i=1:numData
    nama = dirImage(i).name;  
    if regexp(nama, '(D|F)-[0-9]{1,2}.jpg')
   
       B = cell(1,2); 
        if regexp(nama, 'D-[0-9]{1,2}.jpg')
      
            B{1,1} = double(imread([folder, '/', nama]));
            B{1,2} = 1; 
        elseif regexp(nama, 'F-[0-9]{1,2}.jpg')
         
            B{1,1} = double(imread([folder, '/', nama]));
            B{1,2} = -1; 
      end
        M = cat(1,M,B); 
    end
end


numDataTrain = size(M,1); 
class = zeros(numDataTrain,1);
arrayImage = zeros(numDataTrain, 300 * 300);

for i=1:numDataTrain
    im = M{i,1} ;
    im = rgb2gray(im); 
    im = imresize(im, [300 300]); 
    im = reshape(im', 1, 300*300); 
    arrayImage(i,:) = im; 
    class(i) = M{i,2}; 
end

SVMStruct = svmtrain(arrayImage, class);



imTest = double(imread(s)); 
imTest = rgb2gray(imTest); 
imTest = imresize(imTest, [300 300]); 
imTest = reshape(imTest',1, 300*300); 
result = svmclassify(SVMStruct, imTest);
if(result==1)
    msgbox('Desert');
else
    msgbox('Forest');
end
function exCode2()
    folder = 'Dataset'; 
dirImage = dir( folder ); 

numData = size(dirImage,1); 

M ={} ; 




for i=1:numData
    nama = dirImage(i).name;  
    if regexp(nama, '(M|R)-[0-9]{1,2}.jpg')
   
       B = cell(1,2); 
        if regexp(nama, 'M-[0-9]{1,2}.jpg')
      
            B{1,1} = double(imread([folder, '/', nama]));
            B{1,2} = 1; 
        elseif regexp(nama, 'R-[0-9]{1,2}.jpg')
         
            B{1,1} = double(imread([folder, '/', nama]));
            B{1,2} = -1; 
      end
        M = cat(1,M,B); 
    end
end


numDataTrain = size(M,1); 
class = zeros(numDataTrain,1);
arrayImage = zeros(numDataTrain, 300 * 300);

for i=1:numDataTrain
    im = M{i,1} ;
    im = rgb2gray(im); 
    im = imresize(im, [300 300]); 
    im = reshape(im', 1, 300*300); 
    arrayImage(i,:) = im; 
    class(i) = M{i,2}; 
end

SVMStruct = svmtrain(arrayImage, class);



imTest = double(imread(s)); 
imTest = rgb2gray(imTest); 
imTest = imresize(imTest, [300 300]); 
imTest = reshape(imTest',1, 300*300); 
result = svmclassify(SVMStruct, imTest);
if(result==1)
    msgbox('Mountain');
else
    msgbox('Residential');
end
function exCode3()
    folder = 'Dataset'; 
dirImage = dir( folder ); 

numData = size(dirImage,1); 

M ={} ; 




for i=1:numData
    nama = dirImage(i).name;  
    if regexp(nama, '(RI|D)-[0-9]{1,2}.jpg')
   
       B = cell(1,2); 
        if regexp(nama, 'RI-[0-9]{1,2}.jpg')
      
            B{1,1} = double(imread([folder, '/', nama]));
            B{1,2} = 1; 
        elseif regexp(nama, 'D-[0-9]{1,2}.jpg')
         
            B{1,1} = double(imread([folder, '/', nama]));
            B{1,2} = -1; 
      end
        M = cat(1,M,B); 
    end
end


numDataTrain = size(M,1); 
class = zeros(numDataTrain,1);
arrayImage = zeros(numDataTrain, 300 * 300);

for i=1:numDataTrain
    im = M{i,1} ;
    im = rgb2gray(im); 
    im = imresize(im, [300 300]); 
    im = reshape(im', 1, 300*300); 
    arrayImage(i,:) = im; 
    class(i) = M{i,2}; 
end

SVMStruct = svmtrain(arrayImage, class);



imTest = double(imread(s)); 
imTest = rgb2gray(imTest); 
imTest = imresize(imTest, [300 300]); 
imTest = reshape(imTest',1, 300*300); 
result = svmclassify(SVMStruct, imTest);
if(result==1)
    msgbox('River');
else
    msgbox('Desert');
end
function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton15.
function pushbutton15_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton16.
function pushbutton16_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str=get(handles.edit1,'String');
I1 = imread(str);
I4 = imadjust(I1,stretchlim(I1));
I5 = imresize(I4,[300,400]);
figure
imshow(I5);title(' Processed Image ');
set(handles.pushbutton1,'enable','off');    
set(handles.pushbutton16,'enable','off');
set(handles.pushbutton12,'enable','on');


% --- Executes on button press in pushbutton17.
function pushbutton17_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
