<html>
<head>
    <title>dit</title>
</head>
<body>
<pre>
    <?php
    # We do not background the call since we need it to finish before building.
    shell_exec('git pull')
    shell_exec('python ../dit/site/build.py ../public_html 5 >> ../build.log 2>&1 &');
    ?>
    Done.
</pre>
</body>
</html>
