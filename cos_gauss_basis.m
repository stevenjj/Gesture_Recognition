function [ phi ] = cos_gauss_basis( h, c, s )
% Returns the result of a gaussian basis function centered at c with width h
%phi = exp(h*cos(s-c)-1);
phi = exp(-h*(cos(s-c)).^2);

end

