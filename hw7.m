%HW7 Homework 7
function [num_questions] = hw7()
format compact;
close all;

%Part-A
%Estimate illuminant color for both the lights
image=imread('color_constancy_images/macbeth_syl-50MR16Q.tif');
figure;
imshow(image);
[lr1,lg1,lb1]=estimate_illuminant_color(image,100,325,150,380);
disp('Illuminant color est for light -> syl-50MR16Q');
l1=[lr1 lg1 lb1];
disp(l1);

image2=imread('color_constancy_images/macbeth_solux-4100.tif');
figure;
imshow(image2);
[lr2,lg2,lb2]=estimate_illuminant_color(image2,100,325,150,380);
disp('Illuminant color est for light -> solux-4100');
l2=[lr2 lg2 lb2];
disp(l2);

%Angular error between illuminant color estimates
angularErr=angular_error(l1,l2);
disp('Angular error between two light colors found above:');
disp(angularErr);

%Diagonal model
D=[lr1/lr2 0 0; 0 lg1/lg2 0; 0 0 lb1/lb2];
newImage=correct_image(D,image2);
newm=max(newImage);
newImage=newImage.*(250/newm);
figure;
imshow(newImage);

m1=max(image);
m2=max(image2);
imageS=image.*(250/m1);
image2S=image2.*(250/m2);
figure;
imshow(imageS);
figure;
imshow(image2S);

%RMS error (r,g)
rms=rms_error(image,image2);
disp(['RMS error(r,g) b/w original & canonical image: ' num2str(rms)]);
rms=rms_error(image,newImage);
disp(['RMS error(r,g) b/w corrected & canonical image: ' num2str(rms)]);

%MaxRGB illumination estimation
report_maxRGB_ae('color_constancy_images/apples2_syl-50MR16Q.tif','color_constancy_images/apples2_solux-4100.tif','apple');
report_maxRGB_ae('color_constancy_images/ball_syl-50MR16Q.tif','color_constancy_images/ball_solux-4100.tif','ball');
report_maxRGB_ae('color_constancy_images/blocks1_syl-50MR16Q.tif','color_constancy_images/blocks1_solux-4100.tif','blocks');

%Macbeth Solus-4100 light Max RGB light estimation
[lmr,lmg,lmb]=max_RGB(image2);
l=[lmr lmg lmb];
disp('Macbeth solus-4100 light Max RGB est. light: ');
disp(l);

%MaxRGB correction
scale_and_display_image('color_constancy_images/apples2_syl-50MR16Q.tif','color_constancy_images/apples2_solux-4100.tif','apple',1);
scale_and_display_image('color_constancy_images/ball_syl-50MR16Q.tif','color_constancy_images/ball_solux-4100.tif','ball',1);
scale_and_display_image('color_constancy_images/blocks1_syl-50MR16Q.tif','color_constancy_images/blocks1_solux-4100.tif','blocks',1);

%GrayWorld illumination estimation
report_gray_world_ae('color_constancy_images/apples2_syl-50MR16Q.tif','color_constancy_images/apples2_solux-4100.tif','apple');
report_gray_world_ae('color_constancy_images/ball_syl-50MR16Q.tif','color_constancy_images/ball_solux-4100.tif','ball');
report_gray_world_ae('color_constancy_images/blocks1_syl-50MR16Q.tif','color_constancy_images/blocks1_solux-4100.tif','blocks');

%GrayWorld correction
scale_and_display_image('color_constancy_images/apples2_syl-50MR16Q.tif','color_constancy_images/apples2_solux-4100.tif','apple',0);
scale_and_display_image('color_constancy_images/ball_syl-50MR16Q.tif','color_constancy_images/ball_solux-4100.tif','ball',0);
scale_and_display_image('color_constancy_images/blocks1_syl-50MR16Q.tif','color_constancy_images/blocks1_solux-4100.tif','blocks',0);

%Part-B
best_diagonal('color_constancy_images/apples2_syl-50MR16Q.tif','color_constancy_images/apples2_solux-4100.tif','apple');
best_diagonal('color_constancy_images/ball_syl-50MR16Q.tif','color_constancy_images/ball_solux-4100.tif','ball');
best_diagonal('color_constancy_images/blocks1_syl-50MR16Q.tif','color_constancy_images/blocks1_solux-4100.tif','blocks');

end

function best_diagonal(origImagePath,canonImagePath,text)
origImage=imread(origImagePath);
canonImage=imread(canonImagePath);
Ro=origImage(:,:,1);
Ro=Ro(:);
Go=origImage(:,:,2);
Go=Go(:);
Bo=origImage(:,:,3);
Bo=Bo(:);

Rc=canonImage(:,:,1);
Rc=Rc(:);
Gc=canonImage(:,:,2);
Gc=Gc(:);
Bc=canonImage(:,:,3);
Bc=Bc(:);

alpha=Ro./Rc;
alpha=median(alpha);
beta=Go./Gc;
beta=median(beta);
gamma=Bo./Bc;
gamma=median(gamma);
D=[alpha 0 0; 0 beta 0; 0 0 gamma];
D=double(D);
newImage=correct_image(D,origImage);
newm=max(newImage);
newImage=newImage.*(250/newm);
mo=max(origImage);
mc=max(canonImage);
origImageS=origImage.*(250/mo);
canonImageS=canonImage.*(250/mc);
figure;
imshow(origImageS);
figure;
imshow(canonImageS);
figure;
imshow(newImage);
rms=rms_error(canonImage,newImage);
disp(['[Best] RMS error(r,g) for ' text ': ' num2str(rms)]);
end

function scale_and_display_image(origImagePath,canonImagePath,text,maxOrGray)
origImage=imread(origImagePath);
canonImage=imread(canonImagePath);
if maxOrGray==1
    [lr1,lg1,lb1]=max_RGB(origImage);
    [lr2,lg2,lb2]=max_RGB(canonImage);
else
    [lr1,lg1,lb1]=gray_world(origImage);
    [lr2,lg2,lb2]=gray_world(canonImage);    
end
D=[lr1/lr2 0 0; 0 lg1/lg2 0; 0 0 lb1/lb2];
newImage=correct_image(D,origImage);
newm=max(newImage);
newImage=newImage.*(250/newm);
mo=max(origImage);
mc=max(canonImage);
origImageS=origImage.*(250/mo);
canonImageS=canonImage.*(250/mc);
figure;
imshow(origImageS);
figure;
imshow(canonImageS);
figure;
imshow(newImage);
rms=rms_error(canonImage,origImage);
disp(['RMS error(r,g) b/n canonical and original image ' text ': ' num2str(rms)]);
rms=rms_error(canonImage,newImage);
disp(['RMS error(r,g) b/n canonical and corrected image ' text ': ' num2str(rms)]);
end

function[newImage]=correct_image(D,image)
newImage=zeros(size(image,1),size(image,2),3);
for i=1:size(image,1)
    for j=1:size(image,2)
        p=image(i,j,:);
        p=double(p);
        p=[p(1,1,1) p(1,1,2) p(1,1,3)];
        p=p';
        newP=D*p;
        newImage(i,j,1)=newP(1,1);
        newImage(i,j,2)=newP(2,1);
        newImage(i,j,3)=newP(3,1);
    end
end
newImage=uint8(newImage);
end

function report_maxRGB_ae(i1_path,i2_path,text)
i1=imread(i1_path);
i2=imread(i2_path);
[lr1,lg1,lb1]=max_RGB(i1);
l1=[lr1 lg1 lb1];
[lr2,lg2,lb2]=max_RGB(i2);
l2=[lr2 lg2 lb2];
ae=angular_error(l1,l2);
disp(['Angular error for ' text ': ' num2str(ae)]);
end

function report_gray_world_ae(i1_path,i2_path,text)
i1=imread(i1_path);
i2=imread(i2_path);
[lr1,lg1,lb1]=gray_world(i1);
l1=[lr1 lg1 lb1];
[lr2,lg2,lb2]=gray_world(i2);
l2=[lr2 lg2 lb2];
ae=angular_error(l1,l2);
disp(['Angular error for ' text ': ' num2str(ae)]);
end

function[r,g,b]=max_RGB(image)
ra=image(:,:,1);
ra=ra(:);
r=double(max(ra));
ga=image(:,:,2);
ga=ga(:);
g=double(max(ga));
ba=image(:,:,3);
ba=ba(:);
b=double(max(ba));
end

function[r,g,b]=gray_world(image)
ra=image(:,:,1);
ra=ra(:);
r=double(2*mean(ra));
ga=image(:,:,2);
ga=ga(:);
g=double(2*mean(ga));
ba=image(:,:,3);
ba=ba(:);
b=double(2*mean(ba));
end

function[angularErr]=angular_error(l1,l2)
l1=double(l1);
l2=double(l2);
X=dot(l1,l2)/(norm(l1)*norm(l2));
angularErr=acosd(X);
end

function[rms]=rms_error(image,newImage)
SquaredErrSum=0;
pixelCount=0;
for i=1:size(image,1)
    for j=1:size(image,2)
        p=image(i,j,:);
        p=[p(1,1,1) p(1,1,2) p(1,1,3)];
        p2=newImage(i,j,:);
        p2=[p2(1,1,1) p2(1,1,2) p2(1,1,3)];
        if((p(1,1)+p(1,2)+p(1,3))>10 && (p2(1,1)+p2(1,2)+p2(1,3))>10)
            r=double(p(1,1)/(p(1,1)+p(1,2)+p(1,3)));
            g=double(p(1,2)/(p(1,1)+p(1,2)+p(1,3)));
            r2=double(p2(1,1)/(p2(1,1)+p2(1,2)+p2(1,3)));
            g2=double(p2(1,2)/(p2(1,1)+p2(1,2)+p2(1,3)));
            e=abs(r2-r)+abs(g2-g);
            SquaredErrSum=SquaredErrSum+(e*e);
            pixelCount=pixelCount+1;
        end
    end
end
rms=sqrt(SquaredErrSum/pixelCount);
end

function[r,g,b]=estimate_illuminant_color(image,x1,y1,x2,y2)
pixelCount=0;
rSum=0;
gSum=0;
bSum=0;
for i=x1:x2
    for j=y1:y2
        p=image(j,i,:);
        p=double(p);
        p=[p(1,1,1) p(1,1,2) p(1,1,3)];
        rSum=rSum+double(p(1,1));
        gSum=gSum+double(p(1,2));
        bSum=bSum+double(p(1,3));
        pixelCount=pixelCount+1;
    end
end
rAvg=double(rSum/pixelCount);
gAvg=double(gSum/pixelCount);
bAvg=double(bSum/pixelCount);
m=max([rAvg; gAvg; bAvg]);
r=double(rAvg*double((250/m)));
g=double(gAvg*double((250/m)));
b=double(bAvg*double((250/m)));
end