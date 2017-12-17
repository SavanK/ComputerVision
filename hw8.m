%HW8 Homework 8
function [num_questions] = hw8()
format compact;
close all;

%Part A
%1. Gradient
image=imread('climber.tiff');
I=rgb2gray(image);
figure;
imshow(I);
dx=[1 -1];
Ix=conv2(I,dx,'same');
m1=max(Ix(:));
Ix=Ix./m1;
figure;
imshow(Ix);

%2. Edge detection
I2=zeros(size(Ix,1),size(Ix,2));
for i=1:size(Ix,1)
    for j=1:size(Ix,2)
        if(Ix(i,j)>0.35)
            I2(i,j)=1;
        end
    end
end
figure;
imshow(I2);

%3. Gaussian mask - smoothing
sigma=2;
val=-1 : 1;
[X Y]=meshgrid(val,val);
guass_filter=exp(-(X.^2+Y.^2)/(2*sigma*sigma));
guass_filter=guass_filter/sum(guass_filter(:));
figure;
surf(X,Y,guass_filter);
Ismooth=conv2(I,guass_filter,'same');
m2=max(Ismooth(:));
Ismooth=Ismooth./m2;
figure;
imshow(Ismooth);

%4. Edge detection in smoothened image
dx=[1 -1];
Ix=conv2(Ismooth,dx,'same');
Ie=zeros(size(Ix,1),size(Ix,2));
for i=1:size(Ix,1)
    for j=1:size(Ix,2)
        if(Ix(i,j)>0.05)
            Ie(i,j)=1;
        end
    end
end
figure;
imshow(Ie);

%5. Edge detection using combined filter
sigma=4;
val=-1 : 1;
[X Y]=meshgrid(val,val);
guass_filter=exp(-(X.^2+Y.^2)/(2*sigma*sigma));
guass_filter=guass_filter/sum(guass_filter(:));
[dx dy]=gradient(guass_filter);
Ix=conv2(conv2(I,dx,'same'),guass_filter,'same');
%Ix=conv2(I,conv2(dx,guass_filter,'same'),'same');
Iy=conv2(conv2(I,dy,'same'),guass_filter,'same');
mx=max(Ix(:));
Ix=Ix./mx;
my=max(Iy(:));
Iy=Ix./my;
Im=sqrt(Ix.*Ix+Iy.*Iy);
figure;
imshow(Im);

%6. Speed check
tic;
val=-1 : 1;
[X Y]=meshgrid(val,val);
gfx=exp(-(X.^2)/(2*sigma*sigma));
gfx=gfx/sum(gfx(:));
gfy=exp(-(Y.^2)/(2*sigma*sigma));
gfy=gfy/sum(gfy(:));
for i=1:1000
    I1=conv2(I,gfx,'same');
    I2=conv2(I1,gfy,'same');
end
t1=toc;
disp(['time 1: ' num2str(t1)]);

tic;
for i=1:1000
    I3=conv2(I,guass_filter,'same');
end
t2=toc;
disp(['time 2: ' num2str(t2)]);

%Part B
Ith=atan2(Iy,Ix);
figure;
imagesc(Ith.*(Im>0.1));
axis image;
colormap(jet);
caxis([0.1 0.9]);
colorbar;

end