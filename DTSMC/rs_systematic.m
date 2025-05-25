function indx=rs_systematic(w,usamp)

%--------------------------------------------------------------------------
% Systematic resampling algorithm
% 
%--------------------------------------------------------------------------
if nargin<2,
    usamp=rand; % random number
end
N=length(w); % number of particles/output indices
indx=zeros(1,N); % preallocate index variable
Q=cumsum(w); % cumulative sum
u=((0:N-1)+usamp)/N; % set strata

j=1;
for i=1:N
    while (Q(j)<u(i))
        j=j+1;
    end
    indx(i)=j;
end
