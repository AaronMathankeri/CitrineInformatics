----------------------------------------------------------------------
-- This file uses a neural network to estimate the stability vector
-- of a compound given two elements
require 'torch'
require 'optim'
require 'nn'
require 'csvigo'
----------------------------------------------------------------------
-- 1. Load the training data -- modified_training_data.csv

-- Split string
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

local filePath = 'feature_training_data.csv'

-- Count number of rows and columns in file
local i = 0
for line in io.lines(filePath) do
  if i == 0 then
    COLS = #line:split(',')
  end
  i = i + 1
end

local ROWS = i - 1  -- Minus 1 because of header

-- Read data from CSV to tensor
local csvFile = io.open(filePath, 'r')
local header = csvFile:read()

local features = torch.Tensor(ROWS, COLS)

local i = 0
for line in csvFile:lines('*l') do
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
    features[i][key] = val
  end
end

csvFile:close()


-- Serialize tensor
--local outputFilePath = 'train.th7'
--torch.save(outputFilePath, features)

-- Deserialize tensor object
--local restored_features = torch.load(outputFilePath)

print("The FEATURES are loaded")
--print(features)
--print(features:size())
----------------------------------------------------------------------
local filePath = 'target_training_data.csv'
--Count number of rows and columns in file
local i = 0
for line in io.lines(filePath) do
  if i == 0 then
    COLS = #line:split(',')
  end
  i = i + 1
end

local ROWS = i 

-- Read data from CSV to tensor
local csvFile = io.open(filePath, 'r')
local header = csvFile:read()

local targetVctrs = torch.Tensor(ROWS, COLS)

local i = 0
for line in csvFile:lines('*l') do
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
    targetVctrs[i][key] = val
  end
end

csvFile:close()
print("The TARGETS are loaded")
--print(#targetVctrs) --equivalent to function below
--print(targetVctrs:size())
--print(targetVctrs)
----------------------------------------------------------------------
-- Take two data frames, create a data set for MLP training


-- leave 10 out for benchmarking
dataSet_size = (targetVctrs:size()[1]) - 11
--print(dataSet_size)

trainData = {}
function trainData:size() return dataSet_size end 


for i=1,trainData:size() do 
   local input= features[i];     
   local output= targetVctrs[i];
   trainData[i] = {input, output};
end

-- use the remaining to benchmark performance of network
testData = {}
trueValues = {}

for i=1,10 do 
   local input= features[trainData:size() + i];     
   local output= targetVctrs[trainData:size() + i];
   testData[i] = {input};
   trueValues[i] = {output};
end
--print(testData[1][1])
--print(trueValues[1][1])

print("The Training Data created...")
----------------------------------------------------------------------
--2. Create the multi-layer perceptron : mlp

mlp = nn.Sequential();  -- make a multi-layer perceptron
inputs = 98; outputs = 11; HUs = 30; -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, HUs))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(HUs, outputs))

-- loss function:
criterion = nn.MSECriterion()


for i = 1,dataSet_size do
   -- random sample
   local input= features[i];     
   local output=targetVctrs[i];

   -- feed it to the neural network and the criterion

   criterion:forward(mlp:forward(input), output)

   -- train over this example in 3 steps

   -- (1) zero the accumulation of the gradients
   mlp:zeroGradParameters()
   -- (2) accumulate gradients
   mlp:backward(input, criterion:backward(mlp.output, output))
   -- (3) update parameters with a 0.01 learning rate
   mlp:updateParameters(0.005)
end

----------------------------
--Test Network:

function clamp(array) 
   for i = 1,11 do
      if array[i] > 0.1
      then array[i] = 1

      else
	 array[i] = 0
      end

   end
   return array 
end
---------------------------- 
local Predictions = {}
for i = 1,10 do
   local current_prediction = mlp:forward(testData[i][1])
   clamp(current_prediction)
--   Predictions[i] = {}
   Predictions[i] = current_prediction
--   print("Prediction :")
--   print(current_prediction)
--
--   print("Actual : ")
--   print(trueValues[i][1])
end

--Write to file
local filename = 'benchmark.txt'
print("Saving to file:",filename )

local MyPredictions = {}
for i=1,10 do
  MyPredictions[i] = {}
   for j=1,11 do
      MyPredictions[i][j] = Predictions[i][j]
   end
end

--csvigo.save{path=filename,data=MyPredictions, header=true}

------------------------------------------------------

local filePath = 'feature_test_data.csv'

-- Count number of rows and columns in file
local i = 0
for line in io.lines(filePath) do
  if i == 0 then
    COLS = #line:split(',')
  end
  i = i + 1
end

local ROWS = i - 1  -- Minus 1 because of header

-- Read data from CSV to tensor
local csvFile = io.open(filePath, 'r')
local header = csvFile:read()

local test_features = torch.Tensor(ROWS, COLS)

local i = 0
for line in csvFile:lines('*l') do
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
    test_features[i][key] = val
  end
end

csvFile:close()


--print(test_features[1])
local Predictions = {}
for i = 1,1000 do
   local current_prediction = mlp:forward(test_features[i])
   clamp(current_prediction)
   Predictions[i] = {}
   Predictions[i] = current_prediction
--   print("Prediction :")
--   print(current_prediction)
----
----   print("Actual : ")
----   print(trueValues[i][1])
end
--
----Write to file
local filename = 'predictions.txt'
print("Saving to file:",filename )

local MyPredictions = {}
for i=1,1000 do
  MyPredictions[i] = {}
   for j=1,11 do
      MyPredictions[i][j] = Predictions[i][j]
   end
end

csvigo.save{path=filename,data=MyPredictions, header=true}
