function F_Int = Simpson1_3rd(Fun,a,b)
n = numel(Fun(1,:)); % It should be odd
h = (b-a)/n;
[E,O] = Even_Odd(n);
for ii=1:numel(Fun(:,1))
    Fun(ii,1) = Fun(ii,1)./2;
    Fun(ii,end) = Fun(ii,end)./2;
    F_Int(ii) = h/3*(4*sum(Fun(ii,E))+2*sum(Fun(ii,O)));
end
end


function [Even,Odd] = Even_Odd(N)
Num = linspace(1,N,N);
Even = [];
Odd = [];
for i1=1:N
    if rem(Num(i1),2)==0
        Even = [Even; Num(i1)];
    else
        Odd = [Odd;Num(i1)];
    end
end
end