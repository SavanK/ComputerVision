%HW10 Homework 10
function [num_questions] = hw10()
format compact;
close all;

%Part A
%Q1
kmeans('sunset.tiff', 5, 0, 0, 0);
kmeans('tiger-1.tiff', 5, 0, 0, 0);
kmeans('tiger-2.tiff', 5, 0, 0, 0);

%Q2
kmeans('sunset.tiff', 5, 0, 1, 0);
kmeans('sunset.tiff', 5, 0, 2, 0);
kmeans('sunset.tiff', 5, 0, 6, 0);
kmeans('sunset.tiff', 5, 0, 10, 0);

%Q3
T=texture_features('sunset.tiff', 2);

kmeans('sunset.tiff', 5, 0, 0, 1, T);
kmeans('sunset.tiff', 5, 0, 1, 1, T);

%Part B
texton_texture();
end

function texton_texture()
train_imgs=['tigers_small/train_small/108004.tiff' 
    'tigers_small/train_small/108009.tiff' 
    'tigers_small/train_small/108014.tiff' 
    'tigers_small/train_small/108019.tiff'
	'tigers_small/train_small/108024.tiff' 
    'tigers_small/train_small/108029.tiff' 
    'tigers_small/train_small/108034.tiff' 
    'tigers_small/train_small/108039.tiff' 
    'tigers_small/train_small/108044.tiff' 
    'tigers_small/train_small/108049.tiff' 
    'tigers_small/train_small/108054.tiff' 
    'tigers_small/train_small/108059.tiff' 
    'tigers_small/train_small/108064.tiff' 
    'tigers_small/train_small/108069.tiff' 
    'tigers_small/train_small/108074.tiff' 
    'tigers_small/train_small/108079.tiff' 
    'tigers_small/train_small/108084.tiff' 
    'tigers_small/train_small/108089.tiff' 
    'tigers_small/train_small/108094.tiff' 
    'tigers_small/train_small/108099.tiff'];

k=6;
thres=0.5;

FR=[];
for i=1:size(train_imgs,1)
    I=imread(train_imgs(i,:));
    I=rgb2gray(I);
    gaborBank=gabor(4,[0 45 90 135 180]);
    I=imgaborfilt(I,gaborBank);

    [row,col,channel]=size(I);
    I=reshape(I,[row*col,5]);
    FR=[FR; I];
end

textons=kmeans_filter(FR, k, thres);

test_path='tigers_small/test_small/108028.tiff';
test=imread(test_path);
[row,col,channel]=size(test);
test=rgb2gray(test);
gaborBank=gabor(4,[0 45 90 135 180]);
test=imgaborfilt(test,gaborBank);

test_seg=zeros(row,col,1);
for i=1:row
    for j=1:col
        filt_resp=test(i,j,:);
        filt_resp=[filt_resp(1,1,1) filt_resp(1,1,2) filt_resp(1,1,3) filt_resp(1,1,4) filt_resp(1,1,5)];
        test_seg(i,j)=which_cluster_2(textons, k, filt_resp);
    end
end

w=3;
test_hist=zeros(row,col,k);
for i=1+w:row-w
    for j=1+w:col-w
        texton_count=zeros(k,1);
        for l=-w:w
            for m=-w:w
                texton_count(test_seg(i+l,j+m),1)=texton_count(test_seg(i+l,j+m),1)+1;
            end
        end
        for q=1:k
            test_hist(i,j,q)=texton_count(q,1);
        end
    end
end

kmeans_2(test_path, 5, 0, 0, test_hist, 0);
kmeans_2(test_path, 5, 0, 0, test_hist, 1);
kmeans_2(test_path, 5, 0, 1, test_hist, 1);
end

function [c_filt] = kmeans_filter(FR, k, thres)
[row,channel]=size(FR);

c=zeros(k,1);
c_filt=zeros(k,5);
for i=1:k
    c(i,1)=randi([1,row]);
    filt_resp=FR(c(i,1),:);
    c_filt(i,1)=filt_resp(1);
    c_filt(i,2)=filt_resp(2);
    c_filt(i,3)=filt_resp(3);
    c_filt(i,4)=filt_resp(4);
    c_filt(i,5)=filt_resp(5);
end

condition=1;

while condition==1
    imageS=zeros(row,1);
    for i=1:row
        filt_resp=FR(i,:);
        filt_resp=[filt_resp(1) filt_resp(2) filt_resp(3) filt_resp(4) filt_resp(5)];
        imageS(i,1)=which_cluster_2(c_filt, k, filt_resp);
    end

    c_filt_new=zeros(k,5);
    c_filt_new=double(c_filt_new);
    c_count=zeros(k,1);
    for i=1:row
        c_filt_new(imageS(i,1),1)=c_filt_new(imageS(i,1),1)+double(FR(i,1));
        c_filt_new(imageS(i,1),2)=c_filt_new(imageS(i,1),2)+double(FR(i,2));
        c_filt_new(imageS(i,1),3)=c_filt_new(imageS(i,1),3)+double(FR(i,3));
        c_filt_new(imageS(i,1),4)=c_filt_new(imageS(i,1),4)+double(FR(i,4));
        c_filt_new(imageS(i,1),5)=c_filt_new(imageS(i,1),5)+double(FR(i,5));
        c_count(imageS(i,1),1)=c_count(imageS(i,1),1)+1;
    end

    for i=1:k
        c_filt_new(i,1)=c_filt_new(i,1)/c_count(i,1);
        c_filt_new(i,2)=c_filt_new(i,2)/c_count(i,1);
        c_filt_new(i,3)=c_filt_new(i,3)/c_count(i,1);
        c_filt_new(i,4)=c_filt_new(i,4)/c_count(i,1);
        c_filt_new(i,5)=c_filt_new(i,5)/c_count(i,1);
    end

    c_obj=zeros(k,1);
    for i=1:k
        c_obj(i,1)=sqrt(objective_func_2(c_filt(i,:),c_filt_new(i,:)));
    end

    c_filt=c_filt_new;
    condition=0;
    for i=1:k
        if c_obj(i,1) > thres
            condition=1;
        end
    end
end

end

function [It]=texture_features(file_path, w)
I=imread(file_path);
I=rgb2gray(I);
I=double(I);

dx=[-1 0 1; -1 0 1; -1 0 1];
dy=dx';
I=conv2(conv2(I, dx, 'same'), dy, 'same');

h=3;
sigma=0.1;
g=fspecial('gaussian', h, sigma);
Is1=conv2(I, g, 'same');

sigma=0.3;
g=fspecial('gaussian', h, sigma);
Is2=conv2(I, g, 'same');

sigma=0.5;
g=fspecial('gaussian', h, sigma);
Is3=conv2(I, g, 'same');

sigma=0.7;
g=fspecial('gaussian', h, sigma);
Is4=conv2(I, g, 'same');

It=zeros(size(I,1),size(I,2));
for i=1+w:size(I,1)-w
    for j=1+w:size(I,2)-w
        ws1=Is1(i-w:i+w,j-w:j+w).^2;
        ws2=Is2(i-w:i+w,j-w:j+w).^2;
        ws3=Is3(i-w:i+w,j-w:j+w).^2;
        ws4=Is4(i-w:i+w,j-w:j+w).^2;
        ws=[ws1(:) ws2(:) ws3(:) ws4(:)];
        It(i,j)=mean(ws(:));
    end
end
It=uint8(It);

figure;
imshow(It);
title([file_path ' - w=' num2str(w)]);

end

function kmeans(file_path, k, thres, lambda, texture_enabled, T)
image=imread(file_path);
[row,col,channel]=size(image);

c=zeros(k,2);
c_color=zeros(k,6);
for i=1:k
    c(i,1)=randi([1,row]);
    c(i,2)=randi([1,col]);
    color=image(c(i,1),c(i,2),:);
    c_color(i,1)=color(1,1,1);
    c_color(i,2)=color(1,1,2);
    c_color(i,3)=color(1,1,3);
    x=(c(i,1)-1)*255/(row-1);
    y=(c(i,2)-1)*255/(col-1);    
    c_color(i,4)=x*lambda;
    c_color(i,5)=y*lambda;
    if texture_enabled==1
        c_color(i,6)=T(c(i,1), c(i,2));
    end
end

condition=1;

while condition==1
    imageS=zeros(row,col);
    for i=1:row
        for j=1:col
            pixel_color=image(i,j,:);
            x=(i-1)*255/(row-1);
            y=(j-1)*255/(col-1);
            if texture_enabled==1
                pixel_color=[pixel_color(1,1,1) pixel_color(1,1,2) pixel_color(1,1,3) double(x*lambda) double(y*lambda) T(i,j)];
            else
                pixel_color=[pixel_color(1,1,1) pixel_color(1,1,2) pixel_color(1,1,3) double(x*lambda) double(y*lambda) 0];
            end
            imageS(i,j)=which_cluster(c_color, k, pixel_color);
        end
    end

    c_color_new=zeros(k,6);
    c_color_new=double(c_color_new);
    c_count=zeros(k,1);
    for i=1:row
        for j=1:col
            x=(i-1)*255/(row-1);
            y=(j-1)*255/(col-1);
            c_color_new(imageS(i,j),1)=c_color_new(imageS(i,j),1)+double(image(i,j,1));
            c_color_new(imageS(i,j),2)=c_color_new(imageS(i,j),2)+double(image(i,j,2));
            c_color_new(imageS(i,j),3)=c_color_new(imageS(i,j),3)+double(image(i,j,3));
            c_color_new(imageS(i,j),4)=c_color_new(imageS(i,j),4)+double(x*lambda);
            c_color_new(imageS(i,j),5)=c_color_new(imageS(i,j),5)+double(y*lambda);
            if texture_enabled==1
                c_color_new(imageS(i,j),6)=c_color_new(imageS(i,j),6)+double(T(i,j));
            else
                c_color_new(imageS(i,j),6)=c_color_new(imageS(i,j),6)+double(0);
            end
            c_count(imageS(i,j),1)=c_count(imageS(i,j),1)+1;
        end
    end

    for i=1:k
        c_color_new(i,1)=c_color_new(i,1)/c_count(i,1);
        c_color_new(i,2)=c_color_new(i,2)/c_count(i,1);
        c_color_new(i,3)=c_color_new(i,3)/c_count(i,1);
        c_color_new(i,4)=uint8(c_color_new(i,4)/c_count(i,1));
        c_color_new(i,5)=uint8(c_color_new(i,5)/c_count(i,1));
        c_color_new(i,6)=uint8(c_color_new(i,6)/c_count(i,1));
    end
    c_color_new=uint8(c_color_new);

    c_obj=zeros(k,1);
    for i=1:k
        c_obj(i,1)=sqrt(objective_func(c_color(i,:),c_color_new(i,:)));
    end

    c_color=c_color_new;
    condition=0;
    for i=1:k
        if c_obj(i,1) > thres
            condition=1;
        end
    end
end

imageSeg=zeros(row,col,3);
for i=1:row
    for j=1:col
        imageSeg(i,j,1)=c_color(imageS(i,j),1);
        imageSeg(i,j,2)=c_color(imageS(i,j),2);
        imageSeg(i,j,3)=c_color(imageS(i,j),3);
    end
end
imageSeg=uint8(imageSeg);

I=zeros(row,col*2,3);
for i=1:row
    for j=1:col
        I(i,j,1)=image(i,j,1);
        I(i,j,2)=image(i,j,2);
        I(i,j,3)=image(i,j,3);
    end
end

for i=1:row
    for j=1:col
        I(i,col+j,1)=imageSeg(i,j,1);
        I(i,col+j,2)=imageSeg(i,j,2);
        I(i,col+j,3)=imageSeg(i,j,3);
    end
end

I=uint8(I);
figure;
imshow(I);
title([file_path ' - k=' num2str(k) ', lambda=' num2str(lambda) ' and texture=' num2str(texture_enabled)]);

end

function kmeans_2(file_path, k, thres, lambda, hist, color_enabled)
image=imread(file_path);
[row,col,channel]=size(image);

c=zeros(k,2);
c_color=zeros(k,11);
for i=1:k
    c(i,1)=randi([1,row]);
    c(i,2)=randi([1,col]);
    vec=hist(c(i,1),c(i,2),:);
    c_color(i,1)=vec(1,1,1);
    c_color(i,2)=vec(1,1,2);
    c_color(i,3)=vec(1,1,3);
    c_color(i,4)=vec(1,1,4);
    c_color(i,5)=vec(1,1,5);
    c_color(i,6)=vec(1,1,6);

    if color_enabled==1
        color=image(c(i,1),c(i,2),:);
        c_color(i,7)=color(1,1,1);
        c_color(i,8)=color(1,1,2);
        c_color(i,9)=color(1,1,3);
    end
    
    x=(c(i,1)-1)*255/(row-1);
    y=(c(i,2)-1)*255/(col-1);    
    c_color(i,10)=x*lambda;
    c_color(i,11)=y*lambda;
end

condition=1;

while condition==1
    imageS=zeros(row,col);
    for i=1:row
        for j=1:col
            pixel_color=image(i,j,:);
            pixel_vec=hist(i,j,:);
            x=(i-1)*255/(row-1);
            y=(j-1)*255/(col-1);

            if color_enabled
                pixel=[pixel_vec(1,1,1) pixel_vec(1,1,2) pixel_vec(1,1,3) pixel_vec(1,1,4) pixel_vec(1,1,5) pixel_vec(1,1,6) pixel_color(1,1,1) pixel_color(1,1,2) pixel_color(1,1,3) double(x*lambda) double(y*lambda)];
            else
                pixel=[pixel_vec(1,1,1) pixel_vec(1,1,2) pixel_vec(1,1,3) pixel_vec(1,1,4) pixel_vec(1,1,5) pixel_vec(1,1,6) 0 0 0 double(x*lambda) double(y*lambda)];                
            end
            imageS(i,j)=which_cluster_3(c_color, k, pixel);
        end
    end

    c_color_new=zeros(k,11);
    c_color_new=double(c_color_new);
    c_count=zeros(k,1);
    for i=1:row
        for j=1:col
            x=(i-1)*255/(row-1);
            y=(j-1)*255/(col-1);
            c_color_new(imageS(i,j),1)=c_color_new(imageS(i,j),1)+double(hist(i,j,1));
            c_color_new(imageS(i,j),2)=c_color_new(imageS(i,j),2)+double(hist(i,j,2));
            c_color_new(imageS(i,j),3)=c_color_new(imageS(i,j),3)+double(hist(i,j,3));
            c_color_new(imageS(i,j),4)=c_color_new(imageS(i,j),4)+double(hist(i,j,4));
            c_color_new(imageS(i,j),5)=c_color_new(imageS(i,j),5)+double(hist(i,j,5));
            c_color_new(imageS(i,j),6)=c_color_new(imageS(i,j),6)+double(hist(i,j,6));

            if color_enabled==1
                c_color_new(imageS(i,j),7)=c_color_new(imageS(i,j),7)+double(image(i,j,1));
                c_color_new(imageS(i,j),8)=c_color_new(imageS(i,j),8)+double(image(i,j,2));
                c_color_new(imageS(i,j),9)=c_color_new(imageS(i,j),9)+double(image(i,j,3));
            else 
                c_color_new(imageS(i,j),7)=c_color_new(imageS(i,j),7)+double(0);
                c_color_new(imageS(i,j),8)=c_color_new(imageS(i,j),8)+double(0);
                c_color_new(imageS(i,j),9)=c_color_new(imageS(i,j),9)+double(0);
            end
            c_color_new(imageS(i,j),10)=c_color_new(imageS(i,j),10)+double(x*lambda);
            c_color_new(imageS(i,j),11)=c_color_new(imageS(i,j),11)+double(y*lambda);

            c_count(imageS(i,j),1)=c_count(imageS(i,j),1)+1;
        end
    end

    for i=1:k
        c_color_new(i,1)=c_color_new(i,1)/c_count(i,1);
        c_color_new(i,2)=c_color_new(i,2)/c_count(i,1);
        c_color_new(i,3)=c_color_new(i,3)/c_count(i,1);
        c_color_new(i,4)=c_color_new(i,4)/c_count(i,1);
        c_color_new(i,5)=c_color_new(i,5)/c_count(i,1);
        c_color_new(i,6)=c_color_new(i,6)/c_count(i,1);

        c_color_new(i,7)=c_color_new(i,7)/c_count(i,1);
        c_color_new(i,8)=c_color_new(i,8)/c_count(i,1);
        c_color_new(i,9)=c_color_new(i,9)/c_count(i,1);

        c_color_new(i,10)=uint8(c_color_new(i,10)/c_count(i,1));
        c_color_new(i,11)=uint8(c_color_new(i,11)/c_count(i,1));
    end
    c_color_new=uint8(c_color_new);

    c_obj=zeros(k,1);
    for i=1:k
        c_obj(i,1)=sqrt(objective_func_3(c_color(i,:),c_color_new(i,:)));
    end

    c_color=c_color_new;
    condition=0;
    for i=1:k
        if c_obj(i,1) > thres
            condition=1;
        end
    end
end

if color_enabled==0
    for i=1:k
        c_color(i,7)=uint8(255/i);
        c_color(i,8)=uint8(255/i);
        c_color(i,9)=uint8(255/i);
    end
end

imageSeg=zeros(row,col,3);
for i=1:row
    for j=1:col
        imageSeg(i,j,1)=c_color(imageS(i,j),7);
        imageSeg(i,j,2)=c_color(imageS(i,j),8);
        imageSeg(i,j,3)=c_color(imageS(i,j),9);
    end
end
imageSeg=uint8(imageSeg);

I=zeros(row,col*2,3);
for i=1:row
    for j=1:col
        I(i,j,1)=image(i,j,1);
        I(i,j,2)=image(i,j,2);
        I(i,j,3)=image(i,j,3);
    end
end

for i=1:row
    for j=1:col
        I(i,col+j,1)=imageSeg(i,j,1);
        I(i,col+j,2)=imageSeg(i,j,2);
        I(i,col+j,3)=imageSeg(i,j,3);
    end
end

I=uint8(I);
figure;
imshow(I);
title([file_path ' - k=' num2str(k) ', lambda=' num2str(lambda)]);

end

function[cluster]=which_cluster(c, k, pixel)
o=zeros(k,1);
for i=1:k
    c_color=c(i,:);
    o(i,1)=objective_func(c_color,pixel);
end

o_min=min(o);
for i=1:k
    if(o_min==o(i))
        cluster=i;
    end
end
end

function[cluster]=which_cluster_2(c, k, pixel)
o=zeros(k,1);
for i=1:k
    c_color=c(i,:);
    o(i,1)=objective_func_2(c_color,pixel);
end

o_min=min(o);
for i=1:k
    if(o_min==o(i))
        cluster=i;
    end
end
end

function[cluster]=which_cluster_3(c, k, pixel)
o=zeros(k,1);
for i=1:k
    c_color=c(i,:);
    o(i,1)=objective_func_3(c_color,pixel);
end

o_min=min(o);
for i=1:k
    if(o_min==o(i))
        cluster=i;
    end
end
end

function[o_val]=objective_func(c1, c2)
c1=double(c1);
c2=double(c2);
o_val=double((c1(1)-c2(1))^2 + (c1(2)-c2(2))^2 + (c1(3)-c2(3))^2 + (c1(4)-c2(4))^2 + (c1(5)-c2(5))^2 + (c1(6)-c2(6))^2);
end

function[o_val]=objective_func_2(c1, c2)
c1=double(c1);
c2=double(c2);
o_val=double((c1(1)-c2(1))^2 + (c1(2)-c2(2))^2 + (c1(3)-c2(3))^2 + (c1(4)-c2(4))^2 + (c1(5)-c2(5))^2);
end

function[o_val]=objective_func_3(c1, c2)
c1=double(c1);
c2=double(c2);
o_val=double((c1(1)-c2(1))^2 + (c1(2)-c2(2))^2 + (c1(3)-c2(3))^2 + (c1(4)-c2(4))^2 + (c1(5)-c2(5))^2 + (c1(6)-c2(6))^2 + (c1(7)-c2(7))^2 + (c1(8)-c2(8))^2 + (c1(9)-c2(9))^2 + (c1(10)-c2(10))^2 + (c1(11)-c2(11))^2);
end
