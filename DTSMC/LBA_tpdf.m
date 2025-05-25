function f = LBA_tpdf(t, A, b, v, sv)
% Get PDF of first passage time of ith accumulator in LBA model
% F = LBA_tpdf(t, A, b, v, sv)


g = (b-A-t.*v)./(t.*sv);
h = (b-t.*v)./(t.*sv);

g=real(g);
h=real(h);

temp1=normcdf(g);
temp2=normcdf(h);

id=temp1>0.999;
temp1(id,1)=0.999;
id=temp1<0.001;
temp1(id,1)=0.001;

id=temp2>0.999;
temp2(id,1)=0.999;
id=temp2<0.001;
temp2(id,1)=0.001;
f = ((-v.*temp1 + sv.*normpdf(g) + v.*temp2 - sv.*normpdf(h)))./A;