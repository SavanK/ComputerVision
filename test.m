function test()
I = imread('sunset.tiff');
I=rgb2gray(I);
gaborBank = gabor(4,[0 45 90 135 180]);
gaborMag = imgaborfilt(I,gaborBank);
figure
subplot(2,2,1);
for p = 1:4
    subplot(2,2,p)
    imshow(gaborMag(:,:,p),[]);
    theta = gaborBank(p).Orientation;
    lambda = gaborBank(p).Wavelength;
    title(sprintf('Orientation=%d, Wavelength=%d',theta,lambda));
end
end