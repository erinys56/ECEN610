% Histogram data for 4-bit ADC
counts = [43 115 85 101 122 170 75 146 125 60 95 95 115 40 120 242];

% Number of codes
N = length(counts);

% Ideal count value
ideal = sum(counts) / N;

% Calculate DNL
DNL = (counts - ideal) / ideal;

% Calculate INL as cumulative sum of DNL
INL = cumsum(DNL);

% Plotting
figure;
subplot(2,1,1);
stem(0:N-1, DNL, 'filled');
title('DNL');
xlabel('Code');
ylabel('DNL (LSB)');

subplot(2,1,2);
stem(0:N-1, INL, 'filled');
title('INL');
xlabel('Code');
ylabel('INL (LSB)');
