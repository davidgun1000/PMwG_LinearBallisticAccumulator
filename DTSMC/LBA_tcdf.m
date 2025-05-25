function F = LBA_tcdf(t, A, b, v, sv)
% Get CDF of first passage time of ith accumulator in LBA model
% F = LBA_tcdf(t, A, b, v, sv)


g = (b-A-t.*v)./(t.*sv); 
h = (b-t.*v)./(t.*sv);  
i = b-t.*v; 
j = i-A; 

g=real(g);
h=real(h);
i=real(i);
j=real(j);

temp1=real(normcdf(g));
temp2=real(normcdf(h));

id=temp1>0.999;
temp1(id,1)=0.999;
id=temp1<=0.001;
temp1(id,1)=0.001;

id=temp2>0.999;
temp2(id,1)=0.999;
id=temp2<0.001;
temp2(id,1)=0.001;

tmp1 = t.*sv.*(normpdf(g)-normpdf(h));
tmp2 = j.*temp1 - i.*temp2;

F = 1 + (tmp1 + tmp2)./A;