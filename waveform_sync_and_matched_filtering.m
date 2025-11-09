%% Waveform sync and matched filtering with symbol selection
% This script includes the generation process of Tx waveform and
% synchronization between Rx and Tx using autocorrelation. The Tx waveform
% is generated using PRBS and RRC filtering with oversampling factor of 10
% and filter span of 10.
% In our example we applied matched filtering with decimation to select the right
% symbols.

clear; clc; close all;

%% 1. System Parameters (Must match your transmitter)
M = 16;                          % Modulation order
Fs_symbol = 1e9;                 % Symbol Rate
Fs = 10e9;                       % Sample Rate
sps = 10;                        % SamplesPerSymbol (Fs / Fs_symbol)

% RRC Filter Parameters
rolloff = 0.2;                   % RRC rolloff factor
filterSpan = 10;                 % RRC filter span in symbols
numSymbols = 100000;              % Total symbols (preamble + payload)
% --- Generate Preamble ---
preambleLen = 100;
pnPreamble = comm.PNSequence('Polynomial', 'x7+x6+1', ...
    'SamplesPerFrame', log2(M) * preambleLen, ...
    'InitialConditions', ones(1,7));
preambleBits = pnPreamble();
preambleSymbols = qammod(preambleBits, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);

% --- Generate Data Payload ---
payloadLen = numSymbols - preambleLen;
pn = comm.PNSequence('Polynomial', 'x20+x3+1', 'InitialConditions', [zeros(1, 19) 1]);
pn.SamplesPerFrame = log2(M) * payloadLen;
txDataBits = pn(); % Transmitted *payload* bits
txPayloadSymbols = qammod(txDataBits, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
% --- Concatenate and Filter ---
totalTxSymbols = [preambleSymbols; txPayloadSymbols];

rcFilter = comm.RaisedCosineTransmitFilter( ...
    'Shape', 'Square root', ...
    'RolloffFactor', rolloff, ...
    'OutputSamplesPerSymbol', sps, ...
    'FilterSpanInSymbols', filterSpan);
txWaveform = rcFilter(totalTxSymbols);

% Or directly import from the dataset
% txWaveform_i = load('Tx_Waveform/16QAM_i.txt');
% txWaveform_q = load('Tx_Waveform/16QAM_q.txt');
% txWaveform = complex(txWaveform_i, txWaveform_q);




% Import Rx waveform
i_raw = load('1m/16QAM/16QAM_28_7_i.txt');
q_raw = load('1m/16QAM/16QAM_28_7_q.txt');
rxWaveform = complex(i_raw, q_raw);
% Use autocorrelation to find the delay between Rx and Tx
d = finddelay(txWaveform, rxWaveform);
disp(d);
% Shift left for synchronization
rxWaveform = circshift(rxWaveform, -d);

% RRC Matched Filtering
rxRrcFilter = comm.RaisedCosineReceiveFilter(...
    'Shape', 'Square root', ...
    'RolloffFactor', rolloff, ...
    'InputSamplesPerSymbol', sps, ...
    'FilterSpanInSymbols', filterSpan, ...
    'DecimationFactor', 10);
filteredSymbols = rxRrcFilter(rxWaveform);

rxSymbols = filteredSymbols(filterSpan+1:end);
labelSymbols = totalTxSymbols(1:end-filterSpan);

% Export the symbols as complex array if processing in Python
% outputFileName = 'output_symbols.mat';
% dataToSave = [rxSymbols, labelSymbols]; % Concatenate the two arrays
% save(outputFileName, 'dataToSave'); % Save to file in .mat format
