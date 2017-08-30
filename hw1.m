%HW1 Homework 1
%   HW1(infile) takes an image file name from current directory as input.
%   It does the following operations:
%       Q2: Read image data and display image
%       Q3: Write out image to 'out.jpg'
%       Q7:
%           1. Display the information about the variable containing image data.
%           2. Display the size of the image
%           3. Display range of values for Red, Green, Blue and Overall
%           4. Convert the image to grayscale using 'rgb2gray', display the
%           image and print the new image data information
%       Q8:
%           1. Create and display grayscale image based on
%               1. red channel
%               2. green channel
%               3. blue channel
%           2. Circular shift color values to the left and display resultant image
%       Q9:
%           1. Convert grayscale into double precision type.
%           2. Display it using 'imagesc'
%               a. show colorbar for it
%               b. change default colormap(i.e., JET) to gray
%               c. rectify axis to have square pixels rather than distorted pixels
%           3. Use 'imshow' to display image
%       Q10:
%           Set every 5th pixel, horizontally and vertically to 1 i.e., white and
%           display result using 'imagesc' and 'imshow'
%       Q11:
%           Show histogram distribution for each of the color channels (R,G,B)
%       Q12:
%           Plot sin and cos curves over domain [-pi:pi] using 'linspace' and 'plot'
%       Q13:
%           Solve linear equations using -
%               1. 'inv'
%               2. 'linsolve'
%       Q16:
%           1. Set every 5th pixel horizontally and vertically to 0 without using
%           explicit loops i.e., in one simple command and show resulting image.
%           2. Set every pixel with value >0.5 to black and show resulting image.
%       Q17:
%           1. Import data from file and plot
%           2. Find covariance matrix
%           3. "Mean-center" the plot to 
%       Q18:
%           1. Rotate the data and display
%           2. Compare variance before and after transformation
function [num_questions] = hw1(infile)
format compact;
close all;

%Q2 
%   Read image data and display using 'imread' and 'imshow'.
imdata = imread(infile);
figure('Name','Q2 - Image (unedited)','NumberTitle','off');
imshow(imdata);
num_questions=1;

%Q3
%   Write out image using 'imwrite'
num_questions=num_questions+1;
imwrite(imdata, 'out.jpg', 'JPEG');

%Q7
%   1. Display the information about the variable containing image data.
%   2. Display the size of the image
%   3. Display range of values for Red, Green, Blue and Overall
%   4. Convert the image to grayscale using 'rgb2gray', display the
%   image and print the new image data information
num_questions=num_questions+1;
whos imdata;

[num_rows, num_cols, num_channels]=size(imdata);
disp(['Size: ' num2str(num_rows) 'x' num2str(num_cols) 'x' num2str(num_channels)]);

red=imdata(:,:,1);
red_linear=red(:);
disp(['Red range [' num2str(min(red_linear)) ',' num2str(max(red_linear)) ']']);
green=imdata(:,:,2);
green_linear=green(:);
disp(['Green range [' num2str(min(green_linear)) ',' num2str(max(green_linear)) ']']);
blue=imdata(:,:,3);
blue_linear=blue(:);
disp(['Blue range [' num2str(min(blue_linear)) ',' num2str(max(blue_linear)) ']']);
overall=imdata(:);
disp(['Overall range [' num2str(min(overall)) ',' num2str(max(overall)) ']']);

imdata_gray=rgb2gray(imdata);
whos imdata_gray;
figure('Name','Q7 - Black & White Image','NumberTitle','off');
imshow(imdata_gray);

%Q8
%   1. Create and display grayscale image based on
%       1. red channel
%       2. green channel
%       3. blue channel
%   2. Circular shift color values to the left and display resultant image
num_questions=num_questions+1;
figure('Name','Q8 - Red based Grayscale Image','NumberTitle','off');
imshow(red);

figure('Name','Q8 - Green based Grayscale Image','NumberTitle','off');
imshow(green);

figure('Name','Q8 - Blue based Grayscale Image','NumberTitle','off');
imshow(blue);

new_color_img=circshift(imdata,[0 0 -1]);
figure('Name','Q8 - Colors Shifted Image','NumberTitle','off');
imshow(new_color_img);

%Q9
%   1. Convert grayscale into double precision type.
%   2. Display it using 'imagesc'
%       a. show colorbar for it
%       b. change default colormap(i.e., JET) to gray
%       c. rectify axis to have square pixels rather than distorted pixels
%   3. Use 'imshow' to display image
num_questions=num_questions+1;
colormap('JET');
imdata_double=double(imdata);
imdata_double=imdata_double(:,:,:)./255;
imdata_double=rgb2gray(imdata_double);
figure('Name','Q9 - Double precision Grayscale Image','NumberTitle','off');
imagesc(imdata_double);
colorbar;
colormap(gray);
axis image;

figure('Name','Q9 - Double precision Grayscale Image w/ imshow','NumberTitle','off');
imshow(imdata_double);

%Q10
%   Set every 5th pixel, horizontally and vertically to 1 i.e., white and
%   display result using 'imagesc' and 'imshow'
num_questions=num_questions+1;
imdata_double_white=imdata_double;
for R=1:5:num_rows
    for C=1:5:num_cols
        imdata_double_white(R,C)=1;
    end
end
axis auto;
figure('Name','Q10 - White dots w/ imagesc','NumberTitle','off');
imagesc(imdata_double_white);
colormap(gray);
figure('Name','Q10 - White dots w/ imshow','NumberTitle','off');
imshow(imdata_double_white);

%Q11
%   Show histogram distribution for each of the color channels (R,G,B)
num_questions=num_questions+1;
figure('Name','Q11 - Red channel histogram','NumberTitle','off');
histogram(red, 20);
figure('Name','Q11 - Green channel histogram','NumberTitle','off');
histogram(green, 20);
figure('Name','Q11 - Blue channel histogram','NumberTitle','off');
histogram(blue, 20);

%Q12
%   Plot sin and cos curves over domain [-pi:pi] using 'linspace' and 'plot'
num_questions=num_questions+1;
figure('Name','Q12 - Sin/Cos Plot','NumberTitle','off');
x=linspace(-pi,pi,300);
plot(x,sin(x));
hold on;
plot(x,cos(x),'r');

%Q13
%   Solve linear equations using -
%       1. 'inv'
%       2. 'linsolve'
num_questions=num_questions+1;
A=[2 4 1; 2 -1 2; 1 1 -1];
B=[9; 8; 0];
X1=inv(A)*B;
X2=A \ B;
disp(['Linear eqn solved using inverse = ']);
disp(X1);
disp(['Linear eqn solved using linsolve = ']);
disp(X2);
disp(['Difference in above results = ']);
disp(X2-X1);

%Q16
%   1. Set every 5th pixel horizontally and vertically to 0 without using
%   explicit loops i.e., in one simple command and show resulting image.
%   2. Set every pixel with value >0.5 to black and show resulting image.
num_questions=num_questions+1;
imdata_double_black = imdata_double;
imdata_double_black(1:5:num_rows,1:5:num_cols)=0;
figure('Name','Q16 - Black dots','NumberTitle','off');
imshow(imdata_double_black);

imdata_double_r = imdata_double;
imdata_double_r(find(imdata_double_r>0.5))=0;
colormap(gray);
figure('Name','Q16 - Pixels with values >0.5 made black','NumberTitle','off');
imshow(imdata_double_r);

%Q17
%   1. Import data from file and plot
%   2. Find covariance matrix
%   3. "Mean-center" the plot to 
num_questions=num_questions+1;
A=importdata('pca.txt');
X=A(:,1);
Y=A(:,2);
figure('Name','Q17 - X v/s Y plot','NumberTitle','off');
plot(X,Y);
c=cov(A);
disp(['Covariance matrix:']);
disp(c);

xMean=mean(X);
yMean=mean(Y);
figure('Name','Q17 - X v/s Y plot w/ data shifted to mean at origin','NumberTitle','off');
plot(X-xMean,Y-yMean);
ax=gca;
ax.XAxisLocation='origin';
ax.YAxisLocation='origin';

%Q18
%   1. Rotate the data and display
%   2. Compare variance before and after transformation
num_questions=num_questions+1;
X=X-xMean;
Y=Y-yMean;
A=[X Y];
c1=cov(A);
[eV,eD]=eig(c1);
disp(['Proving orthogonality:']);
disp(['eigenVector(eV) Matrix:']);
disp(eV);
disp(['transpose(eV):']);
disp(eV');
disp(['inv(eV):']);
disp(inv(eV));
disp(['diff']);
disp(eV'-inv(eV));
A_rot=A*eV;
figure('Name','Q18 - Mean centered plus rotated','NumberTitle','off');
plot(A_rot(:,1),A_rot(:,2));
xlim([-0.3 0.3]);
ylim([-0.5 0.5]);
ax=gca;
ax.XAxisLocation='origin';
ax.YAxisLocation='origin';

c2=cov(A_rot);
s1=c1(1,1)+c1(2,2);
s2=c2(1,1)+c2(2,2);
disp(['Covariance before transformation:']);
disp(c1);
disp(['Covariance after transformation:']);
disp(c2);
disp(['Sum of variance before transformation:']);
disp(s1);
disp(['Sum of variance after transformation:']);
disp(s2);
