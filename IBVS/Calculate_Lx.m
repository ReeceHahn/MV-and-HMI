function Lx = Calculate_Lx(s,z)
    % Function to calculate the interaction matrix or image Jacobian
    % hint: Eq (11) on the paper or on slide 21 assuming focal length = 1
    % Inputs :
    % s: features
    % z: distance along z axis
    % Output:
    % Lx: interaction matrix/image Jacobian
    % Your code here
    
    n = length(s) / 2;
    Lx = zeros(2*n, 6);

    for i = 1:n
        x = s(2*i - 1);
        y = s(2*i);

        Lx(2*i-1:2*i, :) = [-1/z, 0, x/z, x*y, -(1+(x^2)), y;
                            0, -1/z, y/z, 1 + (y^2), -x*y, -x];
    end
end
