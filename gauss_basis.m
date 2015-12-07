function [ phi ] = gauss_basis( h, c, s )
% Returns the result of a gaussian basis function centered at c with width h
phi = exp(-h*((s-c).^2));

end

