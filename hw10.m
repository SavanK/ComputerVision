%HW10 Homework 10
function [num_questions] = hw10()
format compact;
close all;

%Part A

%Q1
%kmeans('sunset.tiff', 5, 0, 0);
%kmeans('tiger-1.tiff', 5, 0, 0);
%kmeans('tiger-2.tiff', 5, 0, 0);

%Q2
%kmeans('sunset.tiff', 5, 0, 1);
%kmeans('sunset.tiff', 5, 0, 2);
%kmeans('sunset.tiff', 5, 0, 3);
%kmeans('sunset.tiff', 5, 0, 4);
%kmeans('sunset.tiff', 5, 0, 5);
%kmeans('sunset.tiff', 5, 0, 6);
%kmeans('sunset.tiff', 5, 0, 7);
%kmeans('sunset.tiff', 5, 0, 8);
%kmeans('sunset.tiff', 5, 0, 9);
%kmeans('sunset.tiff', 5, 0, 10);

%Q3
T=texture_features('sunset.tiff', 1);
%T=texture_features('sunset.tiff', 2);
%T=texture_features('sunset.tiff', 3);
%T=texture_features('sunset.tiff', 5);

T=T.^(0.5);


end

function [It]=texture_features(file_path, w)
I=imread(file_path);
I=rgb2gray(I);
%figure;
%imshow(I);

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

function kmeans(file_path, k, thres, lambda)
image=imread(file_path);
[row,col,channel]=size(image);

c=zeros(k,2);
c_color=zeros(k,5);
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
end

condition=1;

while condition==1
    imageS=zeros(row,col);
    for i=1:row
        for j=1:col
            pixel_color=image(i,j,:);
            x=(i-1)*255/(row-1);
            y=(j-1)*255/(col-1);
            pixel_color=[pixel_color(1,1,1) pixel_color(1,1,2) pixel_color(1,1,3) double(x*lambda) double(y*lambda)];
            imageS(i,j)=which_cluster(c_color, k, pixel_color);
        end
    end

    c_color_new=zeros(k,5);
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
            c_count(imageS(i,j),1)=c_count(imageS(i,j),1)+1;
        end
    end

    for i=1:k
        c_color_new(i,1)=c_color_new(i,1)/c_count(i,1);
        c_color_new(i,2)=c_color_new(i,2)/c_count(i,1);
        c_color_new(i,3)=c_color_new(i,3)/c_count(i,1);
        c_color_new(i,4)=uint8(c_color_new(i,4)/c_count(i,1));
        c_color_new(i,5)=uint8(c_color_new(i,5)/c_count(i,1));
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
title([file_path ' - k=' num2str(k) ' and lambda=' num2str(lambda)]);

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

function[o_val]=objective_func(c1, c2)
c1=double(c1);
c2=double(c2);
o_val=double((c1(1)-c2(1))^2 + (c1(2)-c2(2))^2 + (c1(3)-c2(3))^2 + (c1(4)-c2(4))^2 + (c1(5)-c2(5))^2);
end