function X = circs()

X = zeros(2,100);

y = 0;
for i = 0:pi/25:2*pi
    y = y + 1;
    X(1, y) = cos(i);
    X(2, y) = sin(i);
end

for i = 0:pi/25:2*pi
    y = y + 1;
    X(1, y) = 2*cos(i);
    X(2, y) = 2*sin(i);
end
