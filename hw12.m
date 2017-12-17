%HW12 Homework 12
function [num_questions] = hw12()
format compact;
close all;
num_questions=0;

%HW 7 - Part A - Q5
hw7_partA_q5();

%HW 7 - Part A - Q6
hw7_partA_q6('color_constancy_images/apples2_syl-50MR16Q.tif','color_constancy_images/apples2_solux-4100.tif','apple');
hw7_partA_q6('color_constancy_images/ball_syl-50MR16Q.tif','color_constancy_images/ball_solux-4100.tif','ball');
hw7_partA_q6('color_constancy_images/blocks1_syl-50MR16Q.tif','color_constancy_images/blocks1_solux-4100.tif','blocks');

%HW 11 - Part E
hw11_partE();

%HW 9 - Part A - Q5
[f1s1 f1s2 f1s3]=hw9_partA_q5('frame1.pgm');
[f2s1 f2s2 f2s3]=hw9_partA_q5('frame2.pgm');
[f3s1 f3s2 f3s3]=hw9_partA_q5('frame3.pgm');

CM=[f1s1 f1s2 f1s3; f2s1 f2s2 f2s3; f3s1 f3s2 f3s3];
disp('Match count confusion matrix: ');
disp(CM);

end

function hw7_partA_q6(origImagePath,canonImagePath,text)
origImage=imread(origImagePath);
canonImage=imread(canonImagePath);
[lr1,lg1,lb1]=max_RGB(origImage);
[lr2,lg2,lb2]=max_RGB(canonImage);
D=[lr1/lr2 0 0; 0 lg1/lg2 0; 0 0 lb1/lb2];
newImage=correct_image(D,origImage);

[lr1,lg1,lb1]=max_RGB(newImage);
l1=[lr1 lg1 lb1];
[lr2,lg2,lb2]=max_RGB(canonImage);
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

function[angularErr]=angular_error(l1,l2)
l1=double(l1);
l2=double(l2);
X=dot(l1,l2)/(norm(l1)*norm(l2));
angularErr=acosd(X);
end

function hw7_partA_q5()
image=imread('color_constancy_images/macbeth_syl-50MR16Q.tif');
[lr1,lg1,lb1]=estimate_illuminant_color(image,100,325,150,380);

image2=imread('color_constancy_images/macbeth_solux-4100.tif');
[lr2,lg2,lb2]=estimate_illuminant_color(image2,100,325,150,380);

%Diagonal model
D=[lr1/lr2 0 0; 0 lg1/lg2 0; 0 0 lb1/lb2];
newImage=correct_image(D,image2);
newm=max(newImage);
newImage=newImage.*(250/newm);

%RMS error (r,g)
rms=rms_error(image,image2);
disp(['RMS error(r,g) b/w original & canonical image: ' num2str(rms)]);
rms=rms_error(image,newImage);
disp(['RMS error(r,g) b/w corrected & canonical image: ' num2str(rms)]);
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
            e=(r2-r)^2+(g2-g)^2;
            SquaredErrSum=SquaredErrSum+e;
            pixelCount=pixelCount+1;
        end
    end
end
rms=sqrt(SquaredErrSum/pixelCount);
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

function [s1_match, s2_match, s3_match]=hw9_partA_q5(frame_pgm)
if1=imread(frame_pgm);
if1=single(if1);
[ff, df] = vl_sift(if1);

is1=imread('slide1.pgm');
is1=single(is1);
[fs1, ds1] = vl_sift(is1);

is2=imread('slide2.pgm');
is2=single(is2);
[fs2, ds2] = vl_sift(is2);

is3=imread('slide3.pgm');
is3=single(is3);
[fs3, ds3] = vl_sift(is3);

euclidean=0;
s1_match=0;
s2_match=0;
s3_match=0;
T=0.85;
for i=1:size(df,2)
    d1 = df(:,i)';
    [n1, n1_dist, x, y] = find_nearest_neighbor(d1, ds1, euclidean);
    [n2, n2_dist, x, y] = find_nearest_neighbor(d1, ds2, euclidean);
    [n3, n3_dist, x, y] = find_nearest_neighbor(d1, ds3, euclidean);
    n_dist=min([n1_dist n2_dist n3_dist]);
    if n_dist <= T
        if n_dist == n1_dist
            s1_match=s1_match+1;
        elseif n_dist == n2_dist
            s2_match=s2_match+1;
        else
            s3_match=s3_match+1;
        end
    end
end
end

function hw11_partE()
im1=imread('S1.jpg');
im2=imread('S2.jpg');
H=homo_ransac('S1.jpg','S2.jpg');

im=zeros(max(size(im1,1), size(im2,1)), size(im1,2)+size(im2,2));
for i=1:size(im1,1)
    for j=1:size(im1,2)
        im(i,j,1)=im1(i,j,1);
        im(i,j,2)=im1(i,j,2);
        im(i,j,3)=im1(i,j,3);
    end
end

for i=1:size(im2,1)
    for j=1:size(im2,2)
        m=map([i,j,1],H);
        r=uint16(m(1));
        c=uint16(m(2));
        if r > 0 && r <= size(im,1) && c > 0 && c < size(im,2)
            if c+size(im1,2) > size(im1,2)
                im(r,c,1)=im2(i,j,1);
                im(r,c,2)=im2(i,j,2);
                im(r,c,3)=im2(i,j,3);
            else
                im(r,c,1)=mean([im(r,c,1) im2(i,j,1)]);
                im(r,c,2)=mean([im(r,c,2) im2(i,j,2)]);
                im(r,c,3)=mean([im(r,c,3) im2(i,j,3)]);
            end
        end
    end
end
im=uint8(im);

figure;
imshow(im);
end

function [bestH]=homo_ransac(im1, im2)
euclidean=0;
lowe_opt=1;
ratio=0.96;
[match, ff, fs, df, ds]=find_matches(im1, im2, lowe_opt, ratio, euclidean);

data_s=size(match,1);
n=4;
w=0.25;
N=0.25;
p=0.99;
k=ceil(log(1-p)/log(1-w^n));

bestH=[];
best_err=999999;
for i=1:k
    rand_match=randperm(data_s,n);
    left=zeros(n,3);
    right=zeros(n,3);
    for j=1:n
        left(j,1)=fs(2,match(rand_match(j),2));
        left(j,2)=fs(1,match(rand_match(j),2));
        left(j,3)=1;
        right(j,1)=ff(2,match(rand_match(j),1));
        right(j,2)=ff(1,match(rand_match(j),1));
        right(j,3)=1;
    end
    H=homography(left,right,n);
    
    left=zeros(data_s,3);
    right=zeros(data_s,3);
    for j=1:data_s
        left(j,1)=fs(2,j);
        left(j,2)=fs(1,j);
        left(j,3)=1;
        right(j,1)=ff(2,j);
        right(j,2)=ff(1,j);
        right(j,3)=1;
    end
    left_m=map(left,H);
    [err err2]=rms_error_2(right,left_m);
    err_s=zeros(data_s,2);
    for j=1:data_s
        err_s(j,1)=j;
        err_s(j,2)=err2(j);
    end
    sortrows(err_s,2);

    bestN=floor(N*data_s);
    left=zeros(bestN,3);
    right=zeros(bestN,3);
    for j=1:bestN
        left(j,1)=fs(2,match(err_s(j,1),2));
        left(j,2)=fs(1,match(err_s(j,1),2));
        left(j,3)=1;
        right(j,1)=ff(2,match(err_s(j,1),1));
        right(j,2)=ff(1,match(err_s(j,1),1));
        right(j,3)=1;
    end
    H=homography(left,right,bestN);
    left_m=map(left,H);    
    [err err2]=rms_error_2(right,left_m);
    
    if err < best_err
        bestH=H;
        best_err=err;
    end    
end
end

function show_homo_ransac(frame_pgm, frame_color, slide_pgm, slide_color)
bestH=homo_ransac(frame_pgm, slide_pgm);

left=zeros(data_s,3);
right=zeros(data_s,3);
for j=1:data_s
    left(j,1)=fs(2,j);
    left(j,2)=fs(1,j);
    left(j,3)=1;
    right(j,1)=ff(2,j);
    right(j,2)=ff(1,j);
    right(j,3)=1;
end
left_m=map(left,bestH);

im_f=imread(frame_color);
im_s=imread(slide_color);
if slide_color=='slide1.tiff'
    im_s=im_s(:,:,1:3);
elseif slide_color=='slide3.tiff'
    im_s=repmat(im_s,1,1,3);
end
im=zeros(max(size(im_f,1),size(im_s,1)), size(im_f,2)+size(im_s,2));

for i=1:size(im_s,1)
    for j=1:size(im_s,2)
        im(i,j,1)=im_s(i,j,1);
        im(i,j,2)=im_s(i,j,2);
        im(i,j,3)=im_s(i,j,3);
    end
end

for i=1:size(im_f,1)
    for j=1:size(im_f,2)
        im(i,j+size(im_s,2),1)=im_f(i,j,1);
        im(i,j+size(im_s,2),2)=im_f(i,j,2);
        im(i,j+size(im_s,2),3)=im_f(i,j,3);
    end
end

for i=1:5:data_s
    p1=[left(i,1) left(i,2)];
    p2=[left_m(i,1) left_m(i,2)+size(im_s,2)];
    im=draw_line(im, p1, p2, 1, 255, 0, 0);    
end
im=uint8(im);

figure;
imshow(im);

end

function map_points(im1_s, im1_f, left1, right1)
H1=homography(left1, right1, 4);
left1_m=map(left1, H1);

im1 = zeros(max([size(im1_s,1) size(im1_f,1)]),size(im1_s,2) + size(im1_f,2), 3);
for i=1:size(im1_s,1)
    for j=1:size(im1_s,2)
        im1(i,j,1)=im1_s(i,j,1);
        im1(i,j,2)=im1_s(i,j,2);
        im1(i,j,3)=im1_s(i,j,3);
    end
end
for i=1:size(im1_f,1)
    for j=1:size(im1_f,2)
        im1(i,j+size(im1_s,2),1)=im1_f(i,j,1);
        im1(i,j+size(im1_s,2),2)=im1_f(i,j,2);
        im1(i,j+size(im1_s,2),3)=im1_f(i,j,3);
    end
end
im1=uint8(im1);

figure, imagesc(im1), axis image, hold on;
sz=50;
X=left1(:,1);
Y=left1(:,2);
scatter(X,Y,sz,'r','filled');
X=right1(:,1);
X=X+size(im1_s,2);
Y=right1(:,2);
scatter(X,Y,sz,'b','filled');
X=left1_m(:,1);
X=X+size(im1_s,2);
Y=left1_m(:,2);
scatter(X,Y,sz,'g','filled');
end

function err=random_dlt(k)
left=zeros(k,3);
right=zeros(k,3);
for i=1:k
    left(i,1)=rand();
    left(i,2)=rand();
    left(i,3)=1;
    right(i,1)=rand();
    right(i,2)=rand();
    right(i,3)=1;
end

H=homography(left, right, k);
left_m=map(left, H);

[err err2]=rms_error_2(right,left_m);
end

function left_m=map(left, H)
left_m=[];
for i=1:size(left,1)
    temp=(H*left(i,:)')';
    temp=[temp(1,1)/temp(1,3) temp(1,2)/temp(1,3) 1];
    left_m=[left_m; temp];
end
end

function H=homography(left, right, k)
U=[];
for i=1:k
    l=left(i,:);
    r=right(i,:);
    rx=r(1);
    ry=r(2);
    z=[0 0 0];
    
    temp1=[z -l ry.*l];
    temp2=[l z -rx.*l];
    temp3=[-ry.*l rx.*l z];
    
    U=[U; temp1; temp2; temp3];
end

[eV,eD]=eig(U'*U);
H=reshape(eV(:,1), [3 3])';
end

function [lineX,lineY,error]=model(data)
xMean=mean(data(:,1));
yMean=mean(data(:,2));
Ux=data(:,1)-xMean;
Uy=data(:,2)-yMean;
U2=[Ux Uy];
[eV,eD]=eig(U2'*U2);
a=eV(1,1);
b=eV(2,1);
d=a*xMean+b*yMean;
m=-a/b;
c=d/b;
lineX=[min(data(:,1)),max(data(:,1))];
lineY=m.*lineX+c;

dLineVals=ones(size(data,1),1).*d;
dPointsVals=a*data(:,1)+b*data(:,2);
[error]=rms_error_1(dLineVals,dPointsVals);
end

function d=distance_point_to_line(pt, p1, p2)
a=p1-p2;
b=pt-p2;
a=[a 1];
b=[b 1];
d=norm(cross(a,b))/norm(a);
end

function [rms_error] = rms_error_1(A_line,A_points)
A_diff=(A_line-A_points).^2;
rms_error=sqrt(mean(A_diff));
end

function [rms_error, err] = rms_error_2(A,B)
err=zeros(size(A,1),1);
for i=1:size(A,1)
    d=A(i,:)-B(i,:);
    d=d.^2;
    d=sqrt(sum(d(:)));
    err(i,1)=d;
end
err=err.^2;
rms_error=sqrt(mean(err(:)));
end

function [match, ff, fs, df, ds]=find_matches(frame_pgm, slide_pgm, lowe_opt, ratio, euclidean)
if1=imread(frame_pgm);
if1=rgb2gray(if1);
if1=single(if1);
[ff, df] = vl_sift(if1);

is1=imread(slide_pgm);
is1=rgb2gray(is1);
is1=single(is1);
[fs, ds] = vl_sift(is1);

match = zeros(size(df,2), 3);
for i=1:size(df,2)
    d1 = df(:,i)';
    [neighbor, n_dist, second_neighbor, second_n_dist] = find_nearest_neighbor(d1, ds, euclidean);
    if(lowe_opt==1)
        if((n_dist/second_n_dist) <= ratio)
            match(i,1) = i;
            match(i,2) = neighbor;
            match(i,3) = n_dist;
        end
    else
        match(i,1) = i;
        match(i,2) = neighbor;
        match(i,3) = n_dist;
    end
end
match = match(any(match,2),:);
match = sortrows(match,3); %3rd column is the distance
end

function [neighbor, n_dist, second_neighbor, second_n_dist] = find_nearest_neighbor(d1, ds, euclidean)
    d2 = ds(:,1)';
    dist1 = distance(d1, d2, euclidean);
    d2 = ds(:,2)';
    dist2 = distance(d1, d2, euclidean);
    if(dist1 < dist2)
        neighbor=1;
        n_dist=dist1;
        second_neighbor=2;
        second_n_dist=dist2;
    else
        neighbor=2;
        n_dist=dist2;
        second_neighbor=1;
        second_n_dist=dist1;
    end
    for j=3:size(ds,2)
        d2 = ds(:,j)';
        dist = distance(d1, d2, euclidean);
        if(dist < n_dist)
            second_neighbor=neighbor;
            second_n_dist=n_dist;
            neighbor=j;
            n_dist=dist;
        elseif(dist < second_n_dist)
            second_n_dist=dist;
            second_neighbor=j;
        end
    end
end

function distance = distance(d1, d2, euclidian)
    if(euclidian == 1)
        distance=euclidian_distance(d1, d2);
    else
        distance=angular_distance(d1, d2);
    end
end

function distance = euclidian_distance(d1, d2)
    diff = d1 - d2;
    diff = diff.^2;
    distance = sqrt(sum(diff));
end

function distance = angular_distance(d1, d2)
    d1=double(d1);
    d2=double(d2);
    distance = acos(dot(d1,d2)/(norm(d1)*norm(d2)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function new_img = draw_line(img, p1, p2, width, r, g, b)

   [ i j ] = bresenham ( p1(1), p1(2), p2(1), p2(2) );
   new_img = draw_points(img, [ i j ], width, r, g, b);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function new_img = draw_points (img, points, box_size, r, g, b)
   new_img = img;

   count = size(points, 1);

   for i = 1:count
       new_img = draw_box (new_img, points(i,1), points(i,2), box_size, r, g, b);
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function new_img = draw_box (img, ci, cj, box_size, r, g, b)

    new_img = img;

    [ m n d ] = size(img);

    i_min = max(1, round(ci) - box_size);
    i_max = min(m, round(ci) + box_size);

    j_min = max(1, round(cj) - box_size);
    j_max = min(n, round(cj) + box_size);

    for i =  i_min:i_max
        for j =  j_min:j_max
            new_img(i, j, 1) = r;
            new_img(i, j, 2) = g;
            new_img(i, j, 3) = b;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Nice code from the web provided by  Aaron Wetzler. 
%

function [x y]=bresenham(x1,y1,x2,y2)

%Matlab optmized version of Bresenham line algorithm. No loops.
%Format:
%               [x y]=bham(x1,y1,x2,y2)
%
%Input:
%               (x1,y1): Start position
%               (x2,y2): End position
%
%Output:
%               x y: the line coordinates from (x1,y1) to (x2,y2)
%
%Usage example:
%               [x y]=bham(1,1, 10,-5);
%               plot(x,y,'or');
x1=round(x1); x2=round(x2);
y1=round(y1); y2=round(y2);
dx=abs(x2-x1);
dy=abs(y2-y1);
steep=abs(dy)>abs(dx);
if steep t=dx;dx=dy;dy=t; end

%The main algorithm goes here.
if dy==0 
    q=zeros(dx+1,1);
else
    q=[0;diff(mod([floor(dx/2):-dy:-dy*dx+floor(dx/2)]',dx))>=0];
end

%and ends here.

if steep
    if y1<=y2 y=[y1:y2]'; else y=[y1:-1:y2]'; end
    if x1<=x2 x=x1+cumsum(q);else x=x1-cumsum(q); end
else
    if x1<=x2 x=[x1:x2]'; else x=[x1:-1:x2]'; end
    if y1<=y2 y=y1+cumsum(q);else y=y1-cumsum(q); end
end

end
