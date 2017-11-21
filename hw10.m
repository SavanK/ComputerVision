%HW10 Homework 10
function [num_questions] = hw10()
format compact;
close all;

%Part A
kmeans('sunset.tiff', 10, 0);
kmeans('tiger-1.tiff', 10, 0);
kmeans('tiger-2.tiff', 10, 0);

end

function kmeans(file_path, k, thres)
image=imread(file_path);
[row,col,channel]=size(image);

c=zeros(k,2);
c_color=zeros(k,3);
for i=1:k
    c(i,1)=randi([1,row]);
    c(i,2)=randi([1,col]);
    color=image(c(i,1),c(i,2),:);
    c_color(i,1)=color(1,1,1);
    c_color(i,2)=color(1,1,2);
    c_color(i,3)=color(1,1,3);
end

condition=1;

while condition==1
    imageS=zeros(row,col);
    for i=1:row
        for j=1:col
            pixel_color=image(i,j,:);
            pixel_color=[pixel_color(1,1,1) pixel_color(1,1,2) pixel_color(1,1,3)];
            imageS(i,j)=which_cluster(c_color, k, pixel_color);
        end
    end

    c_color_new=zeros(k,3);
    c_color_new=double(c_color_new);
    c_count=zeros(k,1);
    for i=1:row
        for j=1:col
            c_color_new(imageS(i,j),1)=c_color_new(imageS(i,j),1)+double(image(i,j,1));
            c_color_new(imageS(i,j),2)=c_color_new(imageS(i,j),2)+double(image(i,j,2));
            c_color_new(imageS(i,j),3)=c_color_new(imageS(i,j),3)+double(image(i,j,3));
            c_count(imageS(i,j),1)=c_count(imageS(i,j),1)+1;
        end
    end

    for i=1:k
        c_color_new(i,1)=c_color_new(i,1)/c_count(i,1);
        c_color_new(i,2)=c_color_new(i,2)/c_count(i,1);
        c_color_new(i,3)=c_color_new(i,3)/c_count(i,1);
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
o_val=double((c1(1)-c2(1))^2 + (c1(2)-c2(2))^2 + (c1(3)-c2(3))^2);
end