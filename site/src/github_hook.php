<html>
<head>
    <title>dit</title>
</head>
<body>
<pre>
    <?php
    shell_exec('python ../dit/site/build.py ../public_html 5 >> ../build.log 2>&1 &');
    ?>
    Done.
</pre>
</body>
</html>
