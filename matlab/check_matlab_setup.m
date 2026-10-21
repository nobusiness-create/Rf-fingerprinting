
clc;
clear;

fprintf("=== MATLAB Environment Check ===\n\n");

% MATLAB version
disp("MATLAB version:");
disp(version);

% Installed products
disp("Installed products:");
ver

% Check deep-learning functions
fprintf("\n=== Deep Learning Toolbox Check ===\n");

if license("test", "Neural_Network_Toolbox")
    disp("A neural-network toolbox license is available.");
else
    disp("A neural-network toolbox license was not detected.");
end

fprintf("\nChecking CNN training functions:\n");

disp("trainNetwork:");
which trainNetwork

disp("trainnet:");
which trainnet

fprintf("\nEnvironment check completed.\n");
