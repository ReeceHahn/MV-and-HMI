function Lx = Calculate_Lx(s,z)
    % Function to calculate the interaction matrix or image Jacobian
    % hint: Eq (11) on the paper or on slide 21 assuming focal length = 1
    % Inputs :
    % s: features
    % z: distance along z axis
    % Output:
    % Lx: interaction matrix/image Jacobian
    % Your code here
    Lx = [-1/z, 0, s(1)/z, s(1)*s(2), -(1+(s(1).^2)), s(2);
        0, -1/z, s(2)/z, 1 + (s(2).^2), -s(1)*s(2), -s(1)];
end