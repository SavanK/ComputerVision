%HW4 Homework 4
function [num_questions] = hw4()
format compact;
close all;

%Part-A
%   'Clicked' coordinate system
imdata = imread('tent.jpg');
figure('Name','Part A - "Clicking" Coordinate System','NumberTitle','off');
imdata(56,40,:)=[255 0 0];
imshow(imdata);
num_questions=1;

%Part-B
%   Calculate Camera Matrix
world_coords=importdata('world_coords.txt');
world_coords=[world_coords, ones(15,1)];
image_coords=importdata('image_coords.txt');
grid_data = imread('IMG_0862.jpeg');

P=[];
z=repelem(0,4);
for i=1:15
    wi=world_coords(i,:);
    ii=image_coords(i,:);

    iu=ii(1);
    r1=[wi, z, -iu*wi];
    
    iv=ii(2);
    r2=[z, wi, -iv*wi];
    
    P=[P; r1; r2];
end

TP=P'*P;
[V,D]=eig(TP);
m=V(:,1);

M=reshape(m,4,3)';
m1=M(1,:)';
m2=M(2,:)';
m3=M(3,:)';

U=[];
V=[];
for i=1:15
    wi=world_coords(i,:);
    ui=dot(m1,wi)/dot(m3,wi);
    vi=dot(m2,wi)/dot(m3,wi);
    U=horzcat(U,ui);
    V=horzcat(V,vi);
end

f1=figure('Name','Part B - Actual and Projected points','NumberTitle','off');
ix=image_coords(:,1);
iy=image_coords(:,2);
show_image(f1,grid_data,U,V,ix,iy);

projected=[U;V];
[rms_error]=rms_error(projected,image_coords);
disp(['RMS error: ' num2str(rms_error)]);
num_questions=num_questions+1;

%Part-B
%   Calculate Camera Matrix
M1=[-3.0940610e-02  -5.2329614e-02   1.4036351e-01  -5.0692177e-01; 1.3467343e-01  -5.2158102e-02   2.1602414e-02  -8.3564809e-01; 1.9710213e-05   3.2851852e-05   3.0513165e-05  -9.4623321e-04];
M2=[-2.8855951e-02  -4.5278096e-02   1.2095919e-01  -4.4892190e-01; 1.4493828e-01  -3.5604489e-02   3.5407011e-02  -8.7030221e-01; 3.9341370e-05   3.3505357e-05   3.2016439e-05  -9.2094577e-04];

m11=M1(1,:)';
m12=M1(2,:)';
m13=M1(3,:)';

m21=M2(1,:)';
m22=M2(2,:)';
m23=M2(3,:)';

U1=[];
V1=[];
U2=[];
V2=[];
for i=1:15
    wi=world_coords(i,:);
    ui1=dot(m11,wi)/dot(m13,wi);
    vi1=dot(m12,wi)/dot(m13,wi);
    U1=horzcat(U1,ui1);
    V1=horzcat(V1,vi1);

    ui2=dot(m21,wi)/dot(m23,wi);
    vi2=dot(m22,wi)/dot(m23,wi);
    U2=horzcat(U2,ui2);
    V2=horzcat(V2,vi2);
end

f2=figure('Name','Part B - 2 Camera Matrix projections','NumberTitle','off');
show_image(f2,grid_data,U1,V1,ix,iy);
show_image2(f2,U2,V2);

projected1=[U1;V1];
diff=image_coords'-projected1;
diff=diff(:);
sum=0;
for i=1:15
    sum=sum+diff(i)*diff(i);
end
rms_error1=sqrt(sum/15);
disp(['RMS error Model 1: ' num2str(rms_error1)]);

projected2=[U2;V2];
diff=image_coords'-projected2;
diff=diff(:);
sum=0;
for i=1:15
    sum=sum+diff(i)*diff(i);
end
rms_error2=sqrt(sum/15);
disp(['RMS error Model 2: ' num2str(rms_error2)]);

num_questions=num_questions+1;

end

function show_image(f,grid_data,U,V,ix,iy)
imshow(grid_data);
hold on;
labels = cellstr(num2str((1:15)'));
set(0, 'CurrentFigure', f);
plot(U,V,'b.','MarkerSize',20);
text(U(:), V(:), labels, 'VerticalAlignment','bottom', ...
                             'HorizontalAlignment','right','Color','r');
hold on;
plot(ix,iy,'r.','MarkerSize',20);
text(ix(:), iy(:), labels, 'VerticalAlignment','bottom', ...
                             'HorizontalAlignment','left','Color','b');
hold off;
end

function show_image2(f,U,V)
hold on;
labels = cellstr(num2str((1:15)'));
set(0, 'CurrentFigure', f);
plot(U,V,'g.','MarkerSize',20);
text(U(:), V(:), labels, 'VerticalAlignment','bottom', ...
                             'HorizontalAlignment','right','Color','g');
hold off;
end

function [rms_error]=rms_error(projected,image_coords)
diff=image_coords'-projected;
diff=diff(:);
sum=0;
for i=1:15
    sum=sum+diff(i)*diff(i);
end
rms_error=sqrt(sum/15);
end