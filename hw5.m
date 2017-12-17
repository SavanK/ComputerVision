%HW5 Homework 5
function [num_questions] = hw5()
format compact;
close all;

%Part-A
%   1. Calculate Camera Matrix
%   2. Replot the points
%   3. Calculate RMS error
world_coords=importdata('world_coords.txt');
world_coords=[world_coords, ones(15,1)];
image_coords=importdata('image_coords.txt');
grid_data=imread('IMG_0862.jpeg');

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
disp('Camera Matrix:');
disp(M);

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
%num_questions=num_questions+1;

%Part-B
%   Draw a sphere and shade it
image2=imread('IMG_0861.jpeg');

[phi,theta]=meshgrid(linspace(-pi/2,pi/2,100),linspace(0,2*pi,100));
radius=0.5;
x0=3;y0=2;z0=3;
x=x0+cos(phi).*cos(theta)*radius;
y=y0+cos(phi).*sin(theta)*radius;
z=z0+sin(phi)*radius;
[nx,ny,nz]=surfnorm(x,y,z);

camera_pos=[9,14,11];
points=[x(:),y(:),z(:)];
N=[nx(:),ny(:),nz(:)];
visible_points=[];

for i=1:size(x(:),1)
    a=camera_pos-points(i,:,:);
    if dot(a,N(i,:,:))>0
        visible_points=[visible_points;points(i,:,:)];
    end
end

%Making visible points homogeneous
visible_points=[visible_points,ones(size(visible_points,1),1)];

f2=figure('Name','Sphere projected into the image');
imshow(image2);
hold on;
set(0,'CurrentFigure',f2);
U_sphere=[];
V_sphere=[];

for i=1:size(visible_points,1)
    wi=visible_points(i,:);
    ui=dot(m1,wi)/dot(m3,wi);
    vi=dot(m2,wi)/dot(m3,wi);
    U_sphere=horzcat(U_sphere,ui);
    V_sphere=horzcat(V_sphere,vi);
end

plot(U_sphere,V_sphere,'r.');
hold on;

light=[33,29,44];
shading=[];
for i=1:size(visible_points,1)
    intensity=dot(N(i,:,:),light);
    if intensity<=0
        point=M*(visible_points(i,:,:))';
        point(1:2)=point(1:2)/point(3);
        shading=[shading;point(1), point(2)];
    end
end
plot(shading(:,1),shading(:,2),'k.');
hold off;

%Part-C
%   Calculate the intrinsic and extrinsic parameters from Camera Matrix, M
scaling_const=sqrt(M(3,1)^2+M(3,2)^2+M(3,3)^2);
M=M/scaling_const;
s=sign(M(3,4));

m1=M(1,1:3)';
m2=M(2,1:3)';
m3=M(3,1:3)';

u0=m1'*m3;
v0=m2'*m3;
alpha=sqrt(m1'*m1-u0^2);
beta=sqrt(m2'*m2-v0^2);

K=[alpha,0,u0;0,beta,v0;0,0,1];
disp('Intrinsic parameter matrix, K:');
disp(K);

R=zeros(3,3);
R(3,:)=s*M(3,1:3);
R(1,:)=s*(u0*M(3,1:3)-M(1,1:3))/alpha;
R(2,:)=s*(v0*M(3,1:3)-M(2,1:3))/beta;

T(1)=s*(u0*M(3,4)-M(1,4))/alpha;
T(2)=s*(v0*M(3,4)-M(2,4))/beta;
T(3)=s*M(3,4);
T=T';

[U,D,V] = svd(R);
R=U*V';
X=[R,T];
X=[X;0,0,0,1];
disp('Extrinsic parameter matrix, X:');
disp(X);

projection=[1,0,0,0;0,1,0,0;0,0,1,0];
M_est=K*projection*X;
disp('Estimated Camera Matrix, M_est:');
disp(M_est);

Camera_pos=-R'*T;
Orientation=R'*[0;0;1];
disp('Camera position');
disp(Camera_pos);
disp('Orientation');
disp(Orientation);

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