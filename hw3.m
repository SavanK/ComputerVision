%HW3 Homework 3
%   HW3() 
%       1. Draw scatter plot for data points
%       2. Use Non-homogeneous Linear Least Squares to fit a line
%       3. Use Homogeneous Linear Least Squares to fit a line
%       4. Display [slope,intercept,rms_error] for each method.
function [num_questions] = hw3()
format compact;
close all;
num_questions=0;

%A1
data=importdata('line_data.txt');
figure('Name','A1 line fitting','NumberTitle','off');
scatter(data(:,1),data(:,2),'b');
hold on

%Non-homogeneous Linear Least Squares
[nRows,nCols]=size(data);
U=[];
for r=1:nRows
    U=[U; data(r,1), 1];
end
A=pinv(U)*data(:,2);
m=A(1,1);
c=A(2,1);
lineX=[min(data(:,1)),max(data(:,1))];
lineY=m.*lineX+c;
plot(lineX,lineY,'k');

xLineVals=data(:,1);
yLineVals=m*xLineVals+c;
[rmsError_y]=rms_error(yLineVals,data(:,2));
q=sqrt(m^2+1);
a=-m/q;
b=1/q;
d=c/q;
dLineVals=ones(nRows,1).*d;
dPointsVals=a*data(:,1)+b*data(:,2);
[rmsError_d]=rms_error(dLineVals,dPointsVals);

disp('Non-homogeneous Linear Least Squares -');
display(m,c,rmsError_y,rmsError_d);

%Homogeneous Linear Least Squares
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
plot(lineX,lineY,'r');

dLineVals=ones(nRows,1).*d;
dPointsVals=a*data(:,1)+b*data(:,2);
[rmsError_d]=rms_error(dLineVals,dPointsVals);
xLineVals=data(:,1);
yLineVals=m*xLineVals+c;
[rmsError_y]=rms_error(yLineVals,data(:,2));

disp('Homogeneous Linear Least Squares -');
display((-a/b),(d/b),rmsError_y,rmsError_d);

legend('Data points','Non-homogeneous Linear Least Squares','Homogeneous Linear Least Squares','Location','best');
end

function [rms_error] = rms_error(A_line,A_points)
A_diff=(A_line-A_points).^2;
rms_error=sqrt(mean(A_diff));
end

function display(slope,intercept,error_y,error_d)
disp(['slope:' num2str(slope) ' intercept:' num2str(intercept) ' rms_error_y:' num2str(error_y) ' rms_error_d:' num2str(error_d)]);
end
