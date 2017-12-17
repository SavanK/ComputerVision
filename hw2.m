%HW2 Homework 2
%   HW2() 
%       Q1
%           Display an image that visualizes all 1600 RGB values derived from the
%           given sensor data and random light spectrum generated
%       Q2
%           1. Comparing real sensor values with estimated values.
%               a. dotted line - est values
%               b. dash line - real values
%           2. RMS error between est. sensor and real sensor values for RGB
%           channels.
%           3. RMS error between R,G,B channels calculated using est. sensor and
%           real sensor values.
%       Q3
%           1. Comparing real sensor values with estimated values w/ Gaussian Noise.
%               a. dotted line - est values
%               b. dash line - real values
%           2. Calculating RMS errors for
%               a. Sensors
%               b. RGB responses
%       Q4
%           1. Comparing plots for sd=50 and sd=100 noise values
%           2. Comparing RMS errors for varying sd values
%       Q6
%           1. Estimate sensitivity for real light spectra and RGB responses
%           2. Plot the sensors and calculate RMS errors
%       Q7
%           Use constrained least squares to improve the sensor estimation
%       Q8
%           Smoothening the estimated sensor
function [num_questions] = hw2()
format compact;
close all;
num_questions=0;

%Q1
%   Display an image that visualizes all 1600 RGB values derived from the
%   given sensor data and random light spectrum generated
num_questions=num_questions+1;
S=importdata('rgb_sensors.txt');
rng(477);
L=rand([1600,101]);
C=L*S;
maxC=max(C(:));
C_RGB=C.*(255/maxC);
imdata=reshape(C_RGB(1:1600,:),[40,40,3]);
imdata=repelem(imdata,10,10);
imdata=uint8(imdata);
figure('Name','Q1 - 40x40 Random Light Spectra Image','NumberTitle','off');
imshow(imdata);

%Q2
%   1. Comparing real sensor values with estimated values.
%       a. dotted line - est values
%       b. dash line - real values
%   2. RMS error between est. sensor and real sensor values for RGB
%   channels.
%   3. RMS error between R,G,B channels calculated using est. sensor and
%   real sensor values.
num_questions=num_questions+1;
S_est=(inv(L'*L)*L')*C;
display_plot(S,S_est,'Q2 - Sensors - real v/s estimated plot',0,2.5*(10.^4));

[rmsEr_S_R,rmsEr_S_G,rmsEr_S_B]=rms_error(S,S_est);
text='RMS error between real and estimated sensor - ';
display_rms_error(text,rmsEr_S_R,rmsEr_S_G,rmsEr_S_B);

C_est=L*S_est;
maxC_est=max(C_est(:));
C_RGB_est=C_est.*(255/maxC_est);
[rmsEr_C_R,rmsEr_C_G,rmsEr_C_B]=rms_error(C_RGB,C_RGB_est);
text='RMS error between RGB values calcuated using real and estimated sensor -';
display_rms_error(text,rmsEr_C_R,rmsEr_C_G,rmsEr_C_B);

%Q3
%   1. Comparing real sensor values with estimated values w/ Gaussian Noise.
%       a. dotted line - est values
%       b. dash line - real values
%   2. Calculating RMS errors for
%       a. Sensors
%       b. RGB responses
num_questions=num_questions+1;
sig=10;
t1=['Q3 - Sensors w/ Noise at Noise sd:', num2str(sig)];
t2=['Q3 - Sensors w/ Noise clipped at Noise sd:', num2str(sig)];
[C_n,S_n,C_n_c,S_n_c]=compute_w_noise(1,sig,true,L,S,C,maxC,t1,t2);
[er_Cn_R,er_Cn_G,er_Cn_B]=rms_error(C_RGB,C_n);
[er_Sn_R,er_Sn_G,er_Sn_B]=rms_error(S,S_n);
[er_Cnc_R,er_Cnc_G,er_Cnc_B]=rms_error(C_RGB,C_n_c);
[er_Snc_R,er_Snc_G,er_Snc_B]=rms_error(S,S_n_c);

text='RGB error w/ noise -';
display_rms_error(text,er_Cn_R,er_Cn_G,er_Cn_B);
text='Sensor error w/ noise -';
display_rms_error(text,er_Sn_R,er_Sn_G,er_Sn_B);

text='RGB error w/ noise w/ clipping-';
display_rms_error(text,er_Cnc_R,er_Cnc_G,er_Cnc_B);
text='Sensor error w/ noise w/ clipping -';
display_rms_error(text,er_Snc_R,er_Snc_G,er_Snc_B);

%Q4
%   1. Comparing plots for sd=50 and sd=100 noise values
%   2. Comparing RMS errors for varying sd values
num_questions=num_questions+1;
for i=0:10
    sig=i*10;
    t1=['Q4 - Sensors w/ Noise at Noise sd:', num2str(sig)];
    t2=['Q4 - Sensors w/ Noise clipped at Noise sd:', num2str(sig)];
    [C_n,S_n,C_n_c,S_n_c]=compute_w_noise(1,sig,mod(i,5)==0&&i>0,L,S,C,maxC,t1,t2);
    [er_Cn_R,er_Cn_G,er_Cn_B]=rms_error(C_RGB,C_n);
    [er_Sn_R,er_Sn_G,er_Sn_B]=rms_error(S,S_n);
    [er_Cnc_R,er_Cnc_G,er_Cnc_B]=rms_error(C_RGB,C_n_c);
    [er_Snc_R,er_Snc_G,er_Snc_B]=rms_error(S,S_n_c);
    er_Cn=sqrt(mean([er_Cn_R,er_Cn_G,er_Cn_B].^2));
    er_Sn=sqrt(mean([er_Sn_R,er_Sn_G,er_Sn_B].^2));
    er_Cnc=sqrt(mean([er_Cnc_R,er_Cnc_G,er_Cnc_B].^2));
    er_Snc=sqrt(mean([er_Snc_R,er_Snc_G,er_Snc_B].^2));
    disp(['w/ noise sd:', num2str(sig), ' RGB err:', num2str(er_Cn), ' Sensor err:', num2str(er_Sn)]);
    disp(['w/ noise w/ clipping sd:', num2str(sig), ' RGB err:', num2str(er_Cnc), ' Sensor err:', num2str(er_Snc)]);
end

%Q6
%   1. Estimate sensitivity for real light spectra and RGB responses
%   2. Plot the sensors and calculate RMS errors
num_questions=num_questions+1;
L2=importdata('light_spectra.txt');
C2=importdata('responses.txt');
S_est2=(inv(L2'*L2)*L2')*C2;
display_plot(S,S_est2,'Q6 - (Real light spectra) Sensors - real v/s estimated plot',-2.5*(10.^4),2.5*(10.^4));

[rmsEr_S2_R,rmsEr_S2_G,rmsEr_S2_B]=rms_error(S,S_est2);
text='Sensor RMS error (Real light spectra)- ';
display_rms_error(text,rmsEr_S2_R,rmsEr_S2_G,rmsEr_S2_B);

C_est2=L2*S_est2;
[rmsEr_C2_R,rmsEr_C2_G,rmsEr_C2_B]=rms_error(C2,C_est2);
text='RGB RMS error (Real light spectra)- ';
display_rms_error(text,rmsEr_C2_R,rmsEr_C2_G,rmsEr_C2_B);

rng(477);
L_temp=rand([598,101]);
disp('Condition number, random light spectra');
disp(cond(L_temp));
disp('Condition number, real light spectra');
disp(cond(L2));

%Q7
%   Use constrained least squares to improve the sensor estimation
num_questions=num_questions+1;
H=L2'*L2;
f=-L2'*C2;
r_f=f(:,1);
g_f=f(:,2);
b_f=f(:,3);
A=-eye(101);
b=-zeros(101,1);

r=quadprog(H,r_f,A,b);
g=quadprog(H,g_f,A,b);
b=quadprog(H,b_f,A,b);

S_cls=[r g b];
display_plot(S,S_cls,'Q7 - (Contrained Least Squares) Sensors - real v/s estimated plot',0,2.5*(10.^4));

%Q8
%   Smoothening the estimated sensor
num_questions=num_questions+1;
Q8(L2,C2,S,90);
Q8(L2,C2,S,100);
Q8(L2,C2,S,120);
Q8(L2,C2,S,140);
Q8(L2,C2,S,150);
end

function Q8(L2,C2,S,lambda)
[n_rows,n_cols]=size(L2);
M=zeros(n_cols-1,n_cols);
for r=1:n_cols-1
    M(r,r)=1;
    M(r,r+1)=-1;
end
M=M.*lambda;

%D2=diff(L2(1:101,:));
D2=L2(1:101,:)*M';
L2_aug=[L2; D2'];
C2_aug=[C2; zeros(100,3)];

H=L2_aug'*L2_aug;
f=-L2_aug'*C2_aug;
r_f=f(:,1);
g_f=f(:,2);
b_f=f(:,3);
A=-eye(101);
b=-zeros(101,1);

rSensor=quadprog(H,r_f,A,b,[],[]);
gSensor=quadprog(H,g_f,A,b,[],[]);
bSensor=quadprog(H,b_f,A,b,[],[]);

S_cls_2=[rSensor gSensor bSensor];
display_plot(S,S_cls_2,['Q8 - Lambda=' num2str(lambda)],0,2.5*(10.^4));
end

function [C_RGB_noise,S_est_noise,C_RGB_noise_clip,S_est_noise_clip]=compute_w_noise(mu,sigma,disp_plot,L,S,C,maxC,t1,t2)
GN_RGB=normrnd(mu,(sigma/100)*maxC,[1600,3]);
C_noise=C+GN_RGB;
S_est_noise=(inv(L'*L)*L')*C_noise;
C_RGB_noise=C_noise.*(255/maxC);
if(disp_plot)
    display_plot(S,S_est_noise,t1,-2.5*(10.^4),2.5*(10.^4));
end

C_RGB_noise_clip=C_RGB_noise;
C_RGB_noise_clip(C_RGB_noise_clip>255)=255;
C_RGB_noise_clip(C_RGB_noise_clip<0)=0;
C_noise_clip=C_RGB_noise_clip.*(maxC/255);
S_est_noise_clip=(inv(L'*L)*L')*C_noise_clip;
if(disp_plot)
    display_plot(S,S_est_noise_clip,t2,-2.5*(10.^4),2.5*(10.^4));
end
end

function [rms_r,rms_g,rms_b] = rms_error(A, A_est)
A_diff=(A-A_est).^2;
rms_r=sqrt(mean(A_diff(:,1)));
rms_g=sqrt(mean(A_diff(:,2)));
rms_b=sqrt(mean(A_diff(:,3)));
end

function display_rms_error(text, rms_r, rms_g, rms_b)
disp(text);
disp('Red');
disp(rms_r);
disp('Green');
disp(rms_g);
disp('Blue');
disp(rms_b);
end

function display_plot(A, A_est, text, y_ll, y_ul)
wavelengths=380:4:780;
figure('Name',text,'NumberTitle','off');
plot(wavelengths,A_est(:,1),':r');
hold on
plot(wavelengths,A_est(:,2),':g');
plot(wavelengths,A_est(:,3),':b');
plot(wavelengths,A(:,1),'--r');
plot(wavelengths,A(:,2),'--g');
plot(wavelengths,A(:,3),'--b');
axis([350 780 y_ll y_ul]);
legend('Est. sensor(R)','Est. sensor(G)','Est. sensor(B)','Real sensor(R)','Real sensor(G)','Real sensor(B)');
end