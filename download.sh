DATA_PATH="./data"

#################################
# Glove
# http://nlp.stanford.edu/projects/glove/
# ###############################

GLOVE_PATH="$DATA_PATH/glove"
GLOVE_FNAME="glove.840B.300d"
GLOVE_URL="http://www-nlp.stanford.edu/data/glove.840B.300d.zip"

if [ ! -d $GLOVE_PATH ]
then
    echo " [*] Download Glove dataset..."
    mkdir -p $GLOVE_PATH
    cd $GLOVE_PATH && { curl -O $GLOVE_URL; cd -; }

    unamestr=`uname`
    if [[ "$unamestr" == 'Linux' ]]; then
        unzip "$GLOVE_PATH/$GLOVE_FNAME.zip" -d "$GLOVE_PATH"
    elif [[ "$unamestr" == 'Darwin' ]]; then
        tar -xvf "$GLOVE_PATH/$GLOVE_FNAME.zip" -C "$GLOVE_PATH"
    fi
else
    echo " [*] GLOVE already exists"
fi


#################################
# MCTest
# http://research.microsoft.com/en-us/um/redmond/projects/mctest/
# ###############################

MCTEST_FNAME="MCTest"
MCTEST_FNAME_ANS="MCTestAnswers"
MCTEST_URL="http://research.microsoft.com/en-us/um/redmond/projects/mctest/data/MCTest.zip"
MCTEST_URL_ANS="http://research.microsoft.com/en-us/um/redmond/projects/mctest/data/MCTestAnswers.zip"
MCTEST_RTE="http://research.microsoft.com/en-us/um/redmond/projects/mctest/data/Statements.zip"
MCTEST_RTE_statement="Statements"

if [ ! -d $DATA_PATH ]
then
    echo " [*] make data directory"
    mkdir -p $DATA_PATH
fi

cd $DATA_PATH

echo " [*] Download MCTest dataset..."
if [ -d $MCTEST_FNAME ]
then
    echo " [*] MCTest already exists"
else
    { curl -O $MCTEST_URL; }
    unzip "$MCTEST_FNAME.zip"
    { curl -O $MCTEST_URL_ANS; }
    unzip "$MCTEST_FNAME_ANS.zip"
    { curl -O $MCTEST_RTE; }
    unzip "$MCTEST_RTE_Statements"
fi

##################################
# Facebook babi task data v1.2
# https://research.facebook.com/research/babi/
# ################################

BABI_FNAME="tasks_1-20_v1-2"
BABI_URL="http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"

echo " [*] Download babi task dataset..."
if [ -d $BABI_FNAME ]
then
    echo " [*] babi already exists"
else
    { curl -O $BABI_URL; }
    tar -xvf "$BABI_FNAME.tar.gz"
fi

##################################
# To get MovieQA data, follow the README.md
# http://movieqa.cs.toronto.edu/home/
# ################################

