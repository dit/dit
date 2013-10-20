<html>
<head>
    <title>dit</title>
</head>
<body>
<pre>
<?php
    // We do not background the call since we need it to finish before the
    // new webpage is built. Current directory is: /var/www/dit.io/public_html
    // If this file is messed up somehow, you will likely need to manually
    // update the git repository on the server and then run build.py.
    chdir('../dit');
    shell_exec('date >> ../build.log 2>&1');
    shell_exec('git pull >> ../build.log 2>&1');
    shell_exec('python site/build.py ../public_html 5 >> ../build.log 2>&1 &');
?>

    Build request received.

</pre>
</body>
</html>
