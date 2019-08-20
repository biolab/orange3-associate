if [ $ORANGE == "release" ]; then
    echo "Orange: Skipping separate Orange install"
    return 0
fi

if [ $ORANGE == "master" ]; then
    echo "Orange: from git master"
    pip install https://github.com/biolab/orange3/archive/master.zip
    return $?;
fi

PACKAGE="orange3==$ORANGE"
echo "Orange: installing version $PACKAGE"
pip install $PACKAGE