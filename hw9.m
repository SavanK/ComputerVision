%HW8 Homework 9
function [num_questions] = hw9()
format compact;
close all;

%Part A
P=0.4;
lowe_opt=1;
ratio=0.96;
euclidean=0;
%show_key_points('frame1.pgm', 'frame1.jpg', 'slide1.pgm', 'slide1.tiff', lowe_opt, P, ratio, euclidean);
%show_key_points('frame1.pgm', 'frame1.jpg', 'slide2.pgm', 'slide2.tiff', lowe_opt, P, ratio, euclidean);
%show_key_points('frame1.pgm', 'frame1.jpg', 'slide3.pgm', 'slide3.tiff', lowe_opt, P, ratio, euclidean);

%show_key_points('frame2.pgm', 'frame2.jpg', 'slide1.pgm', 'slide1.tiff', lowe_opt, P, ratio, euclidean);
%show_key_points('frame2.pgm', 'frame2.jpg', 'slide2.pgm', 'slide2.tiff', lowe_opt, P, ratio, euclidean);
%show_key_points('frame2.pgm', 'frame2.jpg', 'slide3.pgm', 'slide3.tiff', lowe_opt, P, ratio, euclidean);

%show_key_points('frame3.pgm', 'frame3.jpg', 'slide1.pgm', 'slide1.tiff', lowe_opt, P, ratio, euclidean);
%show_key_points('frame3.pgm', 'frame3.jpg', 'slide2.pgm', 'slide2.tiff', lowe_opt, P, ratio, euclidean);
%show_key_points('frame3.pgm', 'frame3.jpg', 'slide3.pgm', 'slide3.tiff', lowe_opt, P, ratio, euclidean);

%Part B
window=2;
k=0.04;
sigma=2;
t1=50000000;
t2=5000000;
harris_corner_detector('indoor.jpg', t1, window, k, sigma);
harris_corner_detector('outdoor_natural.jpg', t1, window, k, sigma);
harris_corner_detector('outdoor_city-1.jpg', t2, window, k, sigma);
harris_corner_detector('outdoor_city-2.jpg', t2, window, k, sigma);
harris_corner_detector('outdoor_city-2-rotated.jpg', t2, window, k, sigma);

end

function harris_corner_detector(image_path, threshold, window, k, sigma)
I=imread(image_path);
I=double(rgb2gray(I));

dx = [-1 0 1; -1 0 1; -1 0 1];
dy = dx';
h=2;

Ix = conv2(I, dx, 'same');
Iy = conv2(I, dy, 'same');
g = fspecial('gaussian',h, sigma);

Ix2 = conv2(Ix.^2, g, 'same');
Iy2 = conv2(Iy.^2, g, 'same');
Ixy = conv2(Ix.*Iy, g,'same');

% Found online
%R = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2;

R = zeros(size(I,1), size(I,2));
for i=1+window:size(I,1)-window
    for j=1+window:size(I,2)-window
        h11=mean(mean(Ix2(i-window:i+window,j-window:j+window)));
        h12=mean(mean(Ixy(i-window:i+window,j-window:j+window)));
        h21=mean(mean(Ixy(i-window:i+window,j-window:j+window)));
        h22=mean(mean(Iy2(i-window:i+window,j-window:j+window)));
        H=[h11 h12; h21 h22];
        [V, D]=eig(H);
        lam1=D(1,1);
        lam2=D(2,2);
        R(i,j)=lam1*lam2 - k*((lam1+lam2)/2)^2;
    end
end

radius=6;
sze=2*radius+1;
mx = ordfilt2(R,sze^2,ones(sze));
R = (R==mx)&(R>threshold);

[r,c] = find(R);

figure, imagesc(I), axis image, colormap(gray), hold on;
plot(c,r,'ys'), title(image_path);

end

function show_key_points(frame_pgm, frame_color, slide_pgm, slide_color, lowe_opt, P, ratio, euclidean)
if1=imread(frame_pgm);
if1=single(if1);
[ff df] = vl_sift(if1);
if1_color=imread(frame_color);

for i=1:size(ff,2)
    y=(ff(1,i));
    x=(ff(2,i));
    s=ff(3,i);
    th=ff(4,i);
    p1=[x y];
    p2=[(x+s*cos(th)) (y+s*sin(th))];
    if1_color=draw_line(if1_color, p1, p2, 1, 255, 255, 0);
end

is1=imread(slide_pgm);
is1=single(is1);
[fs ds] = vl_sift(is1);

is1_color=imread(slide_color);
temp = zeros(size(is1_color,1),size(is1_color,2), 3);
for i=1:size(is1_color,1)
    for j=1:size(is1_color,2)
        if(size(is1_color,3) == 1)
            temp(i,j,1)=is1_color(i,j);
            temp(i,j,2)=is1_color(i,j);
            temp(i,j,3)=is1_color(i,j);
        else
            temp(i,j,1)=is1_color(i,j,1);
            temp(i,j,2)=is1_color(i,j,2);
            temp(i,j,3)=is1_color(i,j,3);            
        end
    end
end
is1_color = uint8(temp);

for i=1:size(fs,2)
    y=(fs(1,i));
    x=(fs(2,i));
    s=fs(3,i);
    th=fs(4,i);
    p1=[x y];
    p2=[(x+s*cos(th)) (y+s*sin(th))];
    is1_color=draw_line(is1_color, p1, p2, 1, 255, 255, 0);
end

I = zeros(2*max([size(is1_color,1) size(if1_color,1)]),size(is1_color,2) + size(if1_color,2), 3);
for i=1:size(if1_color,1)
    for j=1:size(if1_color,2)
        I(i,j,1)=if1_color(i,j,1);
        I(i,j,2)=if1_color(i,j,2);
        I(i,j,3)=if1_color(i,j,3);
    end
end
for i=1:size(is1_color,1)
    for j=1:size(is1_color,2)
        I(i,j+size(if1_color,2),1)=is1_color(i,j,1);
        I(i,j+size(if1_color,2),2)=is1_color(i,j,2);
        I(i,j+size(if1_color,2),3)=is1_color(i,j,3);
    end
end
I=uint8(I);

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

disp([frame_color ' -to- ' slide_color ' match count: ' num2str(size(match,1))]);

if1_color=imread(frame_color);
is1_color=imread(slide_color);
temp = zeros(size(is1_color,1),size(is1_color,2), 3);
for i=1:size(is1_color,1)
    for j=1:size(is1_color,2)
        if(size(is1_color,3) == 1)
            temp(i,j,1)=is1_color(i,j);
            temp(i,j,2)=is1_color(i,j);
            temp(i,j,3)=is1_color(i,j);
        else
            temp(i,j,1)=is1_color(i,j,1);
            temp(i,j,2)=is1_color(i,j,2);
            temp(i,j,3)=is1_color(i,j,3);            
        end
    end
end
is1_color = uint8(temp);

Im = zeros(max([size(is1_color,1) size(if1_color,1)]),size(is1_color,2) + size(if1_color,2), 3);
for i=1:size(if1_color,1)
    for j=1:size(if1_color,2)
        Im(i,j,1)=if1_color(i,j,1);
        Im(i,j,2)=if1_color(i,j,2);
        Im(i,j,3)=if1_color(i,j,3);
    end
end
for i=1:size(is1_color,1)
    for j=1:size(is1_color,2)
        Im(i,j+size(if1_color,2),1)=is1_color(i,j,1);
        Im(i,j+size(if1_color,2),2)=is1_color(i,j,2);
        Im(i,j+size(if1_color,2),3)=is1_color(i,j,3);
    end
end
Im=uint8(Im);
Im2=Im;

for i=1:5:size(match,1)
    y1=(ff(1,match(i,1)));
    x1=(ff(2,match(i,1)));
    y2=(size(if1_color,2) + fs(1,match(i,2)));
    x2=(fs(2,match(i,2)));
    p1=[x1 y1];
    p2=[x2 y2];
    Im=draw_line(Im, p1, p2, 1, 255, 0, 0);
end

match = sortrows(match,3); %3rd column is the euclidean distance
for i=1:2:uint8(P*size(match,1))
    y1=(ff(1,match(i,1)));
    x1=(ff(2,match(i,1)));
    y2=(size(if1_color,2) + fs(1,match(i,2)));
    x2=(fs(2,match(i,2)));
    p1=[x1 y1];
    p2=[x2 y2];
    Im2=draw_line(Im2, p1, p2, 1, 255, 0, 0);
end

I2=I;

for i=1:size(Im,1)
    for j=1:size(Im,2)
        I(i+max([size(is1_color,1) size(if1_color,1)]),j,1) = Im(i,j,1);
        I(i+max([size(is1_color,1) size(if1_color,1)]),j,2) = Im(i,j,2);
        I(i+max([size(is1_color,1) size(if1_color,1)]),j,3) = Im(i,j,3);
    end
end

I=uint8(I);
figure;
image(I);
title([frame_color ' v/s ' slide_color ' -- Visualizing every 5th match']);

for i=1:size(Im2,1)
    for j=1:size(Im2,2)
        I2(i+max([size(is1_color,1) size(if1_color,1)]),j,1) = Im2(i,j,1);
        I2(i+max([size(is1_color,1) size(if1_color,1)]),j,2) = Im2(i,j,2);
        I2(i+max([size(is1_color,1) size(if1_color,1)]),j,3) = Im2(i,j,3);
    end
end

I2=uint8(I2);
figure;
image(I2);
title([frame_color ' v/s ' slide_color ' -- Visualizing best ' num2str(P*100) '% matches']);

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
