(function () {
    async function sendPoseReportEmail({ pdf, filename, pose }) {
        if (!window.REPORT_EMAIL_URL) {
            throw new Error("Report email URL is not configured.");
        }

        if (!pdf || typeof pdf.output !== "function") {
            throw new Error("Report PDF could not be prepared.");
        }

        const reportBlob = pdf.output("blob");
        const formData = new FormData();
        formData.append("report", reportBlob, filename || "Sattvalife_Report.pdf");
        formData.append("pose", pose || "Yoga Pose");

        const response = await fetch(window.REPORT_EMAIL_URL, {
            method: "POST",
            body: formData,
            credentials: "same-origin",
        });

        let payload = {};
        try {
            payload = await response.json();
        } catch (error) {
            payload = {};
        }

        if (!response.ok || payload.success === false) {
            throw new Error(payload.error || "Could not send report email.");
        }

        return payload;
    }

    window.sendPoseReportEmail = sendPoseReportEmail;
}());
