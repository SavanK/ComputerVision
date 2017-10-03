%HW6 Homework 6
function [num_questions] = hw6()
format compact;
close all;

%Part-A
i1=imread('4-1.tiff');
i2=imread('4-2.tiff');
i3=imread('4-3.tiff');
i4=imread('4-4.tiff');
i5=imread('4-5.tiff');
i6=imread('4-6.tiff');
i7=imread('4-7.tiff');

lights=importdata('light_directions.txt');
num_lights=size(lights,1);

[rows,cols,channels]=size(i1);
normals=zeros(rows,cols,channels);
intensities=zeros(rows,cols,num_lights);
for k=1:num_lights
    switch k
        case 1
            img=i1;
        case 2
            img=i2;
        case 3
            img=i3;
        case 4
            img=i4;
        case 5
            img=i5;
        case 6
            img=i6;
        case 7
            img=i7;        
    end
    for i=1:rows
        for j=1:cols
            r=img(i,j,1);
            g=img(i,j,2);
            b=img(i,j,3);
            intensities(i,j,k)=norm(double([r g b]));
        end
    end
end

for i=1:rows
    for j=1:cols
        I=intensities(i,j,:);
        I=reshape(I,[num_lights,1]);
        normal=(lights'*lights)\(lights'*I);
        if norm(normal)~=0
            normal=normal/norm(normal);
        else
            normal=[0;0;0];
        end
        normals(i,j,:)=normal;
    end
end

newImage=zeros(rows,cols,channels);
s=[0 0 1];
for i=1:rows
    for j=1:cols
        nx=normals(i,j,1);
        ny=normals(i,j,2);
        nz=normals(i,j,3);
        shading=dot([nx,ny,nz],s);
        c=double(shading);
        if c<0
            c=0;
        end
        newImage(i,j,:)=[c c c];
    end
end

figure('Name','Part A - Image of surface with a light at (0,0,1)');
imshow(newImage);

Z=zeros(rows,cols);
for i=1:rows
    for j=1:cols
        nx=normals(i,j,1);
        ny=normals(i,j,2);
        nz=normals(i,j,3);        
        if i==1 && j==1
            Z(i,j)=0;
        elseif j==1
            Z(i,j)=((-ny+nz*Z(i-1,j))/nz);
        else
            Z(i,j)=((-nx+nz*Z(i,j-1))/nz);            
        end
    end
end

[X,Y]=meshgrid(1:rows,1:cols);
figure('Name','Part A - Surface Map');
surf(X,Y,Z);

%Part B
light_colors=importdata('color_light_colors_1.txt');
light_pos=importdata('color_light_directions_1.txt');
image=imread('color_photometric_stereo_1.tiff');
[X,Y,Z]=recover_surface(light_colors,light_pos,image);
figure('Name','Part B - Surface Map 1');
surf(X,Y,Z);

light_colors=importdata('color_light_colors_2.txt');
light_pos=importdata('color_light_directions_2.txt');
image=imread('color_photometric_stereo_2.tiff');
[X,Y,Z]=recover_surface(light_colors,light_pos,image);
figure('Name','Part B - Surface Map 2');
surf(X,Y,Z);

end

function[X,Y,Z]=recover_surface(light_colors,light_pos,image)
num_lights=size(light_pos,1);
[rows,cols,channels]=size(image);
normals=zeros(rows,cols,channels);
intensities=zeros(rows,cols,3);
lights3=double(zeros(rows,cols,3));

lr=0;
lg=0;
lb=0;
for k=1:num_lights
    x=light_pos(k,1);
    y=light_pos(k,2);
    z=light_pos(k,3);
    r=light_colors(k,1);
    g=light_colors(k,2);
    b=light_colors(k,3);
    p=double(norm([x y z]));
    lr=double(lr+p*r);
    lg=double(lg+p*g);
    lb=double(lb+p*b);
end

for i=1:rows
    for j=1:cols
        r=image(i,j,1);
        g=image(i,j,2);
        b=image(i,j,3);
        intensities(i,j,1)=norm(double(r));
        intensities(i,j,2)=norm(double(g));
        intensities(i,j,3)=norm(double(b));
        lights3(i,j,1)=double(r)/lr;
        lights3(i,j,2)=double(g)/lg;
        lights3(i,j,3)=double(b)/lb;
    end
end

for i=1:rows
    for j=1:cols
        I=intensities(i,j,:);
        I=reshape(I,[3,1]);
        x=lights3(i,j,1);
        y=lights3(i,j,2);
        z=lights3(i,j,3);
        l3=[x; y; z];
        normal=(l3'*l3)\(l3'*I);
        if norm(normal)~=0
            normal=normal/norm(normal);
        else
            normal=[0;0;0];
        end
        normals(i,j,:)=normal;
    end
end

Z=zeros(rows,cols);
for i=1:rows
    for j=1:cols
        nx=normals(i,j,1);
        ny=normals(i,j,2);
        nz=normals(i,j,3);        
        if i==1 && j==1
            Z(i,j)=0;
        elseif j==1
            Z(i,j)=((-ny+nz*Z(i-1,j))/nz);
        else
            Z(i,j)=((-nx+nz*Z(i,j-1))/nz);            
        end
    end
end

[X,Y]=meshgrid(1:rows,1:cols);
end
