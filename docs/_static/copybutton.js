/* Add a [>>>] button on the top-right corner of code samples to hide
 * the >>> and ... prompts and the output and thus make the code
 * copyable.
 */

$(document).ready(function() {

    /* Note, the outer div of code blocks has a class like 'highlight-python'
     * when the specified language was 'python'. It turns out that there are
     * many ways to specify the language, and they can correspond to different
     * lexers. As of now, one cannot determine from the HTML what lexer was
     * used to parse the source code. As an example, Sphinx will autodetect
     * if the code block represents a python console, then the "pycon" lexer
     * is used instead of the "python" lexer, but the class on the outer div
     * might still say 'highlight-python'.
     */
    var div = $('.highlight-python .highlight,' +
                '.highlight-py .highlight,' +
                '.highlight-sage .highlight,' +
                '.highlight-python3 .highlight,' +
                '.highlight-py3 .highlight,' +
                '.highlight-pycon .highlight,' +
                '.highlight-pycon3 .highlight,' +
                /*
                    Note, 'pycon3' is not an alias of any Python lexer found in
                    :mod:`pygements.lexers.agile`.  However, it is provided by
                    :mod:`sphinx.highlighting`.
                */
                '.highlight-pytb .highlight,' +
                '.highlight-py3tb .highlight,' +

                '.highlight-ipython .highlight,' +
                '.highlight-ipython3 .highlight,' +
                '.highlight-ipythontb .highlight,' +
                '.highlight-ipythontb3 .highlight,' +
                '.highlight-ipythoncon .highlight,' +
                '.highlight-ipython3con .highlight,' +
                '.highlight-ipy .highlight,' +
                '.highlight-ipy3 .highlight')


    var pre = div.find('pre');

    // get the styles from the current theme
    pre.parent().parent().css('position', 'relative');
    var hide_text = 'Hide the prompts and output';
    var show_text = 'Show the prompts and output';
    var border_width = pre.css('border-top-width');
    var border_style = pre.css('border-top-style');
    var border_color = pre.css('border-top-color');
    var button_styles = {
        'cursor':'pointer', 'position': 'absolute', 'top': '0', 'right': '0',
        'border-color': border_color, 'border-style': border_style,
        'border-width': border_width, 'color': border_color,
        'padding-left': '0.3em',  'padding-right': '0.3em',
        'border-radius': '0 3px 0 0'
    }

    // create and add the button to all the code blocks that contain >>>
    div.each(function(index) {
        var jthis = $(this);
        if (jthis.find('.gp').length > 0) {
            // TODO: use something more generic that makes sense for IPython.
            var button = $('<i class="copybutton fa fa-eye fa-1g"></i>');
            button.css(button_styles)
            button.attr('title', hide_text);
            jthis.prepend(button);
        }
        // tracebacks (.gt) contain bare text elements that need to be
        // wrapped in a span to work with .nextUntil() (see toggle code)
        jthis.find('pre:has(.gt)').contents()
            .filter(function() {
                return ((this.nodeType == 3) && (this.data.trim().length > 0));
            })
                .wrap('<span>')
            .end();
    });

    /** Define the behavior of the button when it's clicked.
     *
     * Generic.Output    --> .go
     * Generic.Prompt    --> .gp
     * Generic.Heading   --> .gh  (used for Output prompts)
     * Generic.Traceback --> .gt
     *
     * We make elements within traceback invisible rather than hiding them.
     * If we had hid them instead, then the vertical size of the <pre> block
     * would change causing undesirable jumps.  The downside is that when you
     * select all the code to copy it, you will see a weird selection pattern,
     * and the copied code may contain some spaces that were between the
     * <span> elements. Is there a better solution?
     *
     */
    $('.copybutton').toggle(
        function() {
            var button = $(this);
            button.parent().find('.go, .gp, .gh, .gt').hide();
            button.next('pre').find('.gt').nextUntil('.go, .gp, .gh').css('visibility', 'hidden');
            button.attr('class', 'copybutton fa fa-eye-slash fa-1g');
            button.attr('title', show_text);
        },
        function() {
            var button = $(this);
            button.parent().find('.go, .gp, .gh, .gt').show();
            button.next('pre').find('.gt').nextUntil('.gp, .go, .gh').css('visibility', 'visible');
            button.attr('class', 'copybutton fa fa-eye fa-1g');
            button.attr('title', hide_text);
        });
});

